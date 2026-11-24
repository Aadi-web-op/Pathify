import type { Metadata } from "next";
import Link from "next/link";

import { AuthLayout } from "@/components/app/auth-layout";

import { ForgotPasswordForm } from "./forgot-password-form";

export const metadata: Metadata = { title: "Reset your password" };

export default function ForgotPasswordPage() {
  return (
    <AuthLayout
      title="Forgot your password?"
      intro={
        <>
          Enter your email and we&apos;ll send you a link to choose a new one.
          Remembered it?{" "}
          <Link
            href="/signin"
            className="font-semibold text-sign underline underline-offset-4"
          >
            Sign in
          </Link>
        </>
      }
    >
      <ForgotPasswordForm />
    </AuthLayout>
  );
}
