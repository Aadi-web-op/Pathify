import type { Metadata } from "next";
import { Suspense } from "react";

import { AuthLayout } from "@/components/app/auth-layout";

import { ResetPasswordForm } from "./reset-password-form";

export const metadata: Metadata = {
  title: "Choose a new password",
  robots: { index: false },
};

export default function ResetPasswordPage() {
  return (
    <AuthLayout
      title="Choose a new password"
      intro="Pick something you haven't used before. You'll stay signed in here and be signed out everywhere else."
    >
      <Suspense>
        <ResetPasswordForm />
      </Suspense>
    </AuthLayout>
  );
}
