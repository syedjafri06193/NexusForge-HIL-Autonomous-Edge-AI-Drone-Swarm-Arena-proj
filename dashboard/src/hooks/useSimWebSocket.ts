import { useEffect, useRef } from 'react';
import { useNexusStore } from '../store/nexus';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_BASE  = API_BASE.replace(/^http/, 'ws');

export function useSimWebSocket(sessionId: string | null) {
  const { setConnected, setWs, applySnapshot } = useNexusStore();
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout>>();

  useEffect(() => {
    if (!sessionId) return;

    const connect = () => {
      const ws = new WebSocket(`${WS_BASE}/sessions/${sessionId}/ws`);
      wsRef.current = ws;
      setWs(ws);

      ws.onopen = () => {
        setConnected(true);
        clearTimeout(reconnectRef.current);
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          if (data.type !== 'pong') applySnapshot(data);
        } catch {}
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectRef.current = setTimeout(connect, 2000);
      };

      ws.onerror = () => ws.close();
    };

    connect();
    return () => {
      clearTimeout(reconnectRef.current);
      wsRef.current?.close();
    };
  }, [sessionId]);
}
