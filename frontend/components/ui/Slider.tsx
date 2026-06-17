"use client";

import * as RSlider from "@radix-ui/react-slider";

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  suffix = "",
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between text-xs">
        <span className="font-medium text-muted">{label}</span>
        <span className="font-bold text-primary-soft">
          {value}
          {suffix}
        </span>
      </div>
      <RSlider.Root
        className="relative flex h-5 w-full touch-none items-center"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
      >
        <RSlider.Track className="relative h-1.5 grow rounded-full bg-border">
          <RSlider.Range className="absolute h-full rounded-full bg-primary" />
        </RSlider.Track>
        <RSlider.Thumb
          aria-label={label}
          className="block h-4 w-4 rounded-full border-2 border-primary bg-white shadow-md outline-none transition hover:scale-110 focus:ring-2 focus:ring-primary/40"
        />
      </RSlider.Root>
    </div>
  );
}
