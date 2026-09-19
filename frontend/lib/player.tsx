"use client";
// Reproductor único compartido: cualquier componente (transcripción, eventos, gráficas) puede saltar a un instante.
import { createContext, useCallback, useContext, useEffect, useRef, useState, ReactNode } from "react";

type P = {
  time: number; duration: number; playing: boolean; rate: number; volume: number;
  seek: (t: number, play?: boolean) => void; toggle: () => void; setRate: (r: number) => void; setVolume: (v: number) => void;
  playRange: (start: number, end: number) => void;
  audioRef: React.RefObject<HTMLAudioElement | null>;
};
const Ctx = createContext<P>(null as any);
export const usePlayer = () => useContext(Ctx);

export function PlayerProvider({ src, children }: { src: string; children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRateS] = useState(1);
  const [volume, setVolumeS] = useState(1);
  const stopAt = useRef<number | null>(null);

  useEffect(() => {
    const a = audioRef.current;
    if (!a) return;
    let raf = 0;
    const tick = () => {
      setTime(a.currentTime);
      if (stopAt.current != null && a.currentTime >= stopAt.current) { a.pause(); stopAt.current = null; }
      raf = requestAnimationFrame(tick);
    };
    const onPlay = () => { setPlaying(true); raf = requestAnimationFrame(tick); };
    const onPause = () => { setPlaying(false); cancelAnimationFrame(raf); setTime(a.currentTime); };
    const onMeta = () => setDuration(a.duration || 0);
    a.addEventListener("play", onPlay); a.addEventListener("pause", onPause); a.addEventListener("loadedmetadata", onMeta);
    a.addEventListener("durationchange", onMeta); a.addEventListener("seeked", () => setTime(a.currentTime));
    return () => { cancelAnimationFrame(raf); a.removeEventListener("play", onPlay); a.removeEventListener("pause", onPause); a.removeEventListener("loadedmetadata", onMeta); };
  }, [src]);

  const seek = useCallback((t: number, play = true) => {
    const a = audioRef.current; if (!a) return;
    stopAt.current = null;
    a.currentTime = Math.max(0, t); setTime(a.currentTime);
    if (play) a.play().catch(() => {});
  }, []);
  const playRange = useCallback((s: number, e: number) => { seek(s, true); stopAt.current = e; }, [seek]);
  const toggle = useCallback(() => { const a = audioRef.current; if (!a) return; a.paused ? a.play().catch(() => {}) : a.pause(); }, []);
  const setRate = useCallback((r: number) => { setRateS(r); if (audioRef.current) audioRef.current.playbackRate = r; }, []);
  const setVolume = useCallback((v: number) => { setVolumeS(v); if (audioRef.current) audioRef.current.volume = v; }, []);

  return (
    <Ctx.Provider value={{ time, duration, playing, rate, volume, seek, toggle, setRate, setVolume, playRange, audioRef }}>
      <audio ref={audioRef} src={src} preload="metadata" />
      {children}
    </Ctx.Provider>
  );
}
