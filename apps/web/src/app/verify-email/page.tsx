import type { Metadata } from "next";
import { Suspense } from "react";

import { AuthLayout } from "@/components/app/auth-layout";

import { VerifyEmail } from "./verify-email";

export const metadata: Metadata = {
  title: "Confirm your email",
  robots: { index: false },
};

export default function VerifyEmailPage() {
  return (
    <AuthLayout
      title="Confirm your email"
      intro="One moment while we check your link."
    >
      <Suspense>
        <VerifyEmail />
      </Suspense>
    </AuthLayout>
  );
}
