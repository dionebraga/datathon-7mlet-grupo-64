"use client";

import { motion } from "framer-motion";
import { Banknote, CreditCard, Landmark, type LucideIcon, PiggyBank, Shield, XCircle } from "lucide-react";
import { Card, CardTitle } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { brl } from "@/lib/utils";
import type { Offer } from "@/lib/types";

const ICONS: Record<string, LucideIcon> = {
  card: CreditCard,
  credit: Banknote,
  deposit: PiggyBank,
  invest: Landmark,
  insurance: Shield,
  control: XCircle,
};

export function OffersGrid({ offers }: { offers: Offer[] }) {
  return (
    <Card>
      <CardTitle icon={<Landmark className="h-4 w-4 text-primary-soft" />}>
        Catálogo de ofertas ({offers.length} braços)
      </CardTitle>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        {offers.map((o, i) => {
          const Icon = ICONS[o.category] ?? Banknote;
          return (
            <motion.div
              key={o.offer_id}
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.05 }}
              className="rounded-xl border border-border bg-surface2 p-3"
            >
              <div className="flex items-center justify-between">
                <Icon className="h-5 w-5 text-primary-soft" />
                <Badge tone={o.suitability_tier === "restricted" ? "danger" : "muted"}>
                  {o.suitability_tier}
                </Badge>
              </div>
              <div className="mt-2 text-sm font-semibold leading-tight">{o.name}</div>
              <div className="mt-1 flex items-center justify-between text-xs text-muted">
                <span>{o.category}</span>
                <span className="font-bold text-success">{brl(o.margin)}</span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </Card>
  );
}
