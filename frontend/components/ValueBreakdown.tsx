"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export interface OfferValue {
  name: string;
  value: number;
  chosen: boolean;
}

export function ValueBreakdown({ data }: { data: OfferValue[] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(180, data.length * 42)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 28, top: 4, bottom: 4 }}>
        <XAxis type="number" hide />
        <YAxis
          type="category"
          dataKey="name"
          width={150}
          tick={{ fill: "#a1a1aa", fontSize: 12 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.04)" }}
          contentStyle={{
            background: "#0a0a0a",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 12,
            color: "#ededed",
          }}
          formatter={(v: number) => [`R$ ${v.toFixed(1)}`, "valor esperado"]}
        />
        <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={22}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.chosen ? "#34d399" : "#0070f3"} fillOpacity={d.chosen ? 1 : 0.6} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
