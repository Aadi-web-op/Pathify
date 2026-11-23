"use client";

import type { ComponentProps } from "react";

import { ButtonLink } from "@/components/ui/button";
import { burstFrom } from "@/lib/burst";

/** A button-styled link that throws an impact burst as you press it. */
export function BurstLink({ onClick, ...rest }: ComponentProps<typeof ButtonLink>) {
  return (
    <ButtonLink
      {...rest}
      onClick={(e) => {
        burstFrom(e.currentTarget, { confetti: true });
        onClick?.(e);
      }}
    />
  );
}
