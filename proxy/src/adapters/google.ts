import { logger } from '../logger.js';
import type { OpenAIMessage } from '../mapper.js';
import type { StreamChunk, ModelAdapter } from './types.js';
import { poolFetch } from '../http-pool.js';
import { parseToolArgs } from '../utils/parse-tool-args.js';

const AUDIO_MIME_BY_FORMAT: Record<string, string> = {
  wav: 'audio/wav',
  mp3: 'audio/mp3',
  aiff: 'audio/aiff',
  aac: 'audio/aac',
  ogg: 'audio/ogg',
  flac: 'audio/flac',
  m4a: 'audio/mp4',
  pcm16: 'audio/L16',
};

export class GoogleAdapter implements ModelAdapter {
  provider = 'google';
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl: string, apiKey: string) {
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.apiKey = apiKey;
  }

  /**
   * Build the Google streaming endpoint URL.
   * SECURITY: API key is sent in headers (x-goog-api-key), NOT in URL.
   * Keys in URLs leak to logs, HTTP proxies, and referer headers.
   */
  private buildStreamUrl(model: string): string {
    return `${this.baseUrl}/v1beta/models/${model}:streamGenerateContent?alt=sse`;
  }

  /**
   * Build request headers. Includes the API key as x-goog-api-key.
   * Never place the API key in the URL.
   */
  private buildAuthHeaders(): Record<string, string> {
    return {
      'Content-Type': 'application/json',
      'x-goog-api-key': this.apiKey,
    };
  }

  async *stream(
    model: string,
    messages: OpenAIMessage[],
    tools?: Record<string, unknown>,
    config?: Record<string, unknown>,
    signal?: AbortSignal,
    system?: string,
  ): AsyncGenerator<StreamChunk> {
    const body = this.buildRequest(model, messages, tools, config, system);
    const url = this.buildStreamUrl(model);
    const response = await this.fetchResponse(url, body, this.buildAuthHeaders(), signal);

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          // Drain any remaining bytes from decoder and buffer at stream EOF
          buffer += decoder.decode();
          if (buffer.trim()) {
            yield* this.parseSSEChunk(buffer);
          }
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          yield* this.parseSSEChunk(line);
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  private *parseSSEChunk(line: string): Generator<StreamChunk> {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return;
    // Support both 'data: ' and 'data:' per SSE spec
    const data = trimmed.startsWith('data: ') ? trimmed.slice(6).trim() : trimmed.slice(5).trim();
    if (!data || data === '[DONE]') return;
    let chunk: any;
    try { chunk = JSON.parse(data); } catch { return; }
    const candidate = chunk.candidates?.[0];
    if (!candidate) return;
    const parts = candidate.content?.parts || [];
    for (const part of parts) {
      if (part.thought) {
        if (part.text) yield { type: 'thought', content: part.text };
      } else if (part.text) {
        yield { type: 'text', content: part.text };
      }
      if (part.functionCall) {
        const args = typeof part.functionCall.args === 'string'
          ? this.parseToolArgs(part.functionCall.args)
          : (part.functionCall.args || {});
        yield { type: 'tool-call', name: part.functionCall.name, args, id: part.functionCall.id };
      }
    }
    if (candidate.finishReason) {
      yield { type: 'done', finishReason: candidate.finishReason };
    }
  }

  private buildRequest(
    model: string,
    messages: OpenAIMessage[],
    tools?: Record<string, unknown>,
    config?: Record<string, unknown>,
    system?: string,
  ): Record<string, unknown> {
    const body: Record<string, unknown> = {
      contents: this.convertMessages(messages),
      generationConfig: {},
    };
    // Include system instruction as native Gemini system_instruction field
    if (system) {
      (body as any).system_instruction = { parts: [{ text: system }] };
    }
    if (tools && Object.keys(tools).length > 0) {
      (body as any).tools = [{
        functionDeclarations: Object.entries(tools).map(([name, tool]: [string, any]) => ({
          name,
          description: tool.description || '',
          parameters: tool.parameters || { type: 'object', properties: {} },
        })),
      }];
    }
    const gc = body.generationConfig as any;
    if (config?.maxTokens) gc.maxOutputTokens = config.maxTokens;
    if (config?.temperature != null) gc.temperature = config.temperature;
    if (config?.topP != null) gc.topP = config.topP;
    if ((config as any)?.stopSequences?.length) gc.stopSequences = (config as any).stopSequences;
    if (Object.keys(gc).length === 0) delete body.generationConfig;
    return body;
  }

  private convertMessages(messages: OpenAIMessage[]): any[] {
    const result: any[] = [];
    const callNameMap = new Map<string, string>();

    // Pass 1: Pre-populate callNameMap across all messages in context
    for (const m of messages) {
      if (m.tool_calls) {
        for (const tc of m.tool_calls) {
          callNameMap.set(tc.id, tc.function.name);
        }
      }
    }

    for (const m of messages) {
      if (m.role === 'system') continue;

      let role: 'user' | 'model';
      const parts: any[] = [];

      if (m.role === 'tool') {
        role = 'user';
        // Tool Name Resolution: map -> id regex pattern fallback -> unknown
        const toolId = m.tool_call_id || '';
        const regexMatch = toolId.match(/^call_(.+)_\d+$/);
        const name = (m as any).name || callNameMap.get(toolId) || (regexMatch ? regexMatch[1] : 'unknown');

        // Parse response payload into canonical Struct object
        let responseObj: Record<string, unknown>;
        if (typeof m.content === 'object' && m.content !== null && !Array.isArray(m.content)) {
          responseObj = m.content as Record<string, unknown>;
        } else if (typeof m.content === 'string') {
          const trimmed = m.content.trim();
          try {
            const parsed = JSON.parse(trimmed);
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
              responseObj = parsed;
            } else {
              responseObj = { output: parsed ?? m.content };
            }
          } catch {
            responseObj = { output: m.content };
          }
        } else {
          responseObj = { output: '' };
        }

        parts.push({ functionResponse: { name, response: responseObj } });
      } else {
        role = m.role === 'assistant' ? 'model' : 'user';
        // Preserve thinking chain for Gemini models
        if (role === 'model' && m.reasoning_content) {
          parts.push({ thought: true, text: m.reasoning_content });
        }
        if (typeof m.content === 'string' && m.content) parts.push({ text: m.content });
        if (Array.isArray(m.content)) {
          for (const p of m.content) {
            if (p.type === 'image_url' && p.image_url?.url) {
              const url = p.image_url.url;
              const m2 = url.match(/^data:([^;]+);base64,(.+)$/);
              if (m2) {
                parts.push({ inlineData: { mimeType: m2[1], data: m2[2] } });
              } else {
                parts.push({ fileData: { fileUri: url, mimeType: 'image/jpeg' } });
              }
            } else if (p.type === 'input_audio' && p.input_audio?.data) {
              parts.push({
                inlineData: {
                  mimeType: AUDIO_MIME_BY_FORMAT[p.input_audio.format] || 'audio/wav',
                  data: p.input_audio.data,
                },
              });
            } else if (p.type === 'video_url' && p.video_url?.url) {
              const url = p.video_url.url;
              const m2 = url.match(/^data:([^;]+);base64,(.+)$/);
              if (m2) {
                parts.push({ inlineData: { mimeType: m2[1], data: m2[2] } });
              } else if (/^https?:/i.test(url)) {
                parts.push({ fileData: { fileUri: url, mimeType: 'video/mp4' } });
              }
            } else if (p.type === 'text' && p.text) {
              parts.push({ text: p.text });
            } else if (typeof p === 'string') {
              parts.push({ text: p });
            }
          }
        }
        if (m.tool_calls) {
          for (const tc of m.tool_calls) {
            parts.push({
              functionCall: { name: tc.function.name, args: this.parseToolArgs(tc.function.arguments) },
            });
          }
        }
      }

      if (parts.length === 0) continue;

      // Fix Role Alternation Violation: merge consecutive parts with identical role into one turn
      const last = result[result.length - 1];
      if (last && last.role === role) {
        last.parts.push(...parts);
      } else {
        result.push({ role, parts });
      }
    }
    return result;
  }

  private async fetchResponse(url: string, body: Record<string, unknown>, headers: Record<string, string>, signal?: AbortSignal): Promise<Response> {
    const response = await poolFetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal,
    });
    if (!response.ok) {
      const err = await response.text().catch(() => 'unknown');
      throw new Error(`[google] API error ${response.status}: ${err}`);
    }
    return response;
  }

  private parseToolArgs(raw: string): Record<string, unknown> {
    return parseToolArgs(raw);
  }
}
