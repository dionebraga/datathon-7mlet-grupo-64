"use client";

import * as RSelect from "@radix-ui/react-select";
import { Check, ChevronDown } from "lucide-react";

export function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <div className="mb-1.5 text-xs font-medium text-muted">{label}</div>
      <RSelect.Root value={value} onValueChange={onChange}>
        <RSelect.Trigger className="flex w-full items-center justify-between rounded-lg border border-border bg-surface2 px-3 py-2 text-sm text-text outline-none transition hover:border-primary/40 focus:ring-2 focus:ring-primary/30">
          <RSelect.Value />
          <RSelect.Icon>
            <ChevronDown className="h-4 w-4 text-muted" />
          </RSelect.Icon>
        </RSelect.Trigger>
        <RSelect.Portal>
          <RSelect.Content className="z-50 overflow-hidden rounded-lg border border-border bg-surface shadow-2xl">
            <RSelect.Viewport className="p-1">
              {options.map((opt) => (
                <RSelect.Item
                  key={opt}
                  value={opt}
                  className="flex cursor-pointer items-center justify-between rounded-md px-3 py-1.5 text-sm text-text outline-none data-[highlighted]:bg-primary/20"
                >
                  <RSelect.ItemText>{opt}</RSelect.ItemText>
                  <RSelect.ItemIndicator>
                    <Check className="h-3.5 w-3.5 text-primary-soft" />
                  </RSelect.ItemIndicator>
                </RSelect.Item>
              ))}
            </RSelect.Viewport>
          </RSelect.Content>
        </RSelect.Portal>
      </RSelect.Root>
    </div>
  );
}
