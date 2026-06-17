import { cn } from "@/lib/utils";

const tones = {
  primary: "bg-primary/15 text-primary-soft border-primary/25",
  success: "bg-success/15 text-success border-success/25",
  danger: "bg-danger/15 text-danger border-danger/25",
  muted: "bg-white/5 text-muted border-border",
} as const;

export function Badge({
  tone = "primary",
  className,
  children,
}: {
  tone?: keyof typeof tones;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[0.7rem] font-bold",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
