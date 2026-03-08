import { useState, useEffect, useRef } from "react";

export function useRouteSimulator(route) {
  const [currentStopIdx, setCurrentStopIdx] = useState(0);
  const [isPlaying, setIsPlaying]           = useState(false);
  const [agentPos, setAgentPos]             = useState(null);
  const [completedStops, setCompletedStops] = useState(new Set());
  const intervalRef = useRef(null);
  const frameRef    = useRef(null);

  // Reset when a new route is assigned
  useEffect(() => {
    setCurrentStopIdx(0);
    setIsPlaying(false);
    setCompletedStops(new Set());
    setAgentPos(route.length > 0 ? [route[0].lat, route[0].lng] : null);
  }, [route]);

  useEffect(() => {
    if (!isPlaying || route.length === 0) return;

    intervalRef.current = setInterval(() => {
      setCurrentStopIdx(prev => {
        const next = prev + 1;
        if (next >= route.length) {
          setIsPlaying(false);     // journey complete
          clearInterval(intervalRef.current);
          return prev;
        }

        // Animate agent smoothly between current and next stop
        animateMarker(
          [route[prev].lat, route[prev].lng],
          [route[next].lat, route[next].lng],
          setAgentPos
        );

        setCompletedStops(s => new Set([...s, prev]));
        return next;
      });
    }, 2500);   // 2.5 seconds per stop — adjust for demo speed

    return () => clearInterval(intervalRef.current);
  }, [isPlaying, route]);

  function animateMarker(from, to, setter) {
    const steps  = 60;   // animation frames
    const dLat   = (to[0] - from[0]) / steps;
    const dLng   = (to[1] - from[1]) / steps;
    let   step   = 0;

    if (frameRef.current) cancelAnimationFrame(frameRef.current);

    function frame() {
      step++;
      setter([from[0] + dLat * step, from[1] + dLng * step]);
      if (step < steps) frameRef.current = requestAnimationFrame(frame);
    }
    frameRef.current = requestAnimationFrame(frame);
  }

  const play  = () => setIsPlaying(true);
  const pause = () => { setIsPlaying(false); clearInterval(intervalRef.current); };
  const reset = () => {
    clearInterval(intervalRef.current);
    setIsPlaying(false);
    setCurrentStopIdx(0);
    setCompletedStops(new Set());
    setAgentPos(route.length > 0 ? [route[0].lat, route[0].lng] : null);
  };

  return { agentPos, currentStopIdx, completedStops, isPlaying, play, pause, reset };
}
