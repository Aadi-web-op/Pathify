"use client";

import { Button, Spinner } from "@/components/ui/button";
import { Notice } from "@/components/ui/notice";
import { burstFrom } from "@/lib/burst";
import { rupees, useBilling, useBuy } from "@/lib/billing";

/** In Settings: the account's plan, and buying Plus or the school plan. */
export function BillingSection({ isCounsellor }: { isCounsellor: boolean }) {
  const billing = useBilling();
  const buy = useBuy();
  if (billing.isPending) return <Spinner />;
  if (billing.error)
    return <Notice tone="error">{billing.error.message}</Notice>;
  const b = billing.data;
  const until = (d: string | null | undefined) =>
    d
      ? new Date(d).toLocaleDateString("en-IN", {
          day: "numeric",
          month: "long",
          year: "numeric",
        })
      : "";
  const shown = b.plans.filter(
    (p) => p.product === "family_plus" || isCounsellor,
  );

  return (
    <div className="grid gap-4">
      <p className="text-muted">
        Everything a student needs on Pathify is free. Plans add extras for
        families and schools, and keep it free and ad-free for everyone else.
      </p>
      {b.plus && (
        <Notice>
          You have Pathify Plus until {until(b.plus_until)}. Thank you!
        </Notice>
      )}
      {b.school && (
        <Notice>Your school plan runs until {until(b.school_until)}.</Notice>
      )}
      <ul className="grid gap-3 sm:grid-cols-2">
        {shown.map((p) => {
          const owned =
            (p.product === "family_plus" && b.plus) ||
            (p.product === "school" && b.school);
          return (
            <li
              key={p.product}
              className="grid content-between gap-4 rounded-2xl border border-rule p-5"
            >
              <div className="grid gap-2">
                <p className="font-display text-lg font-semibold">{p.name}</p>
                <p className="font-mono text-sm">
                  {rupees(p.amount)} <span className="text-muted">a year</span>
                </p>
                <ul className="grid gap-1 text-sm text-muted">
                  {p.perks.map((perk) => (
                    <li key={perk}>· {perk}</li>
                  ))}
                </ul>
              </div>
              <Button
                variant={owned ? "secondary" : "primary"}
                disabled={!b.configured || buy.isPending}
                busy={buy.isPending && buy.variables === p.product}
                onClick={(e) => {
                  const button = e.currentTarget;
                  buy.mutate(p.product as "family_plus" | "school", {
                    onSuccess: () => burstFrom(button, { confetti: true }),
                  });
                }}
              >
                {owned ? "Add another year" : `Get it for ${rupees(p.amount)}`}
              </Button>
            </li>
          );
        })}
      </ul>
      {!b.configured && (
        <p className="text-sm text-muted">
          Payments aren&apos;t switched on for this server yet.
        </p>
      )}
      {buy.error && buy.error.message !== "Payment cancelled." && (
        <Notice tone="error">{buy.error.message}</Notice>
      )}
    </div>
  );
}
