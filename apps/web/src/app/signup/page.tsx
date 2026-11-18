import type { Metadata } from "next";
import Link from "next/link";

import { AuthLayout } from "@/components/app/auth-layout";

import { SignUpForm } from "./sign-up-form";

export const metadata: Metadata = { title: "Create your account" };

export default function SignUpPage() {
  return (
    <AuthLayout
      title="Create your account"
      intro={
        <>
          Already have one?{" "}
          <Link href="/signin" className="font-semibold text-sign underline underline-offset-4">
            Sign in
          </Link>
        </>
      }
    >
      <SignUpForm />
    </AuthLayout>
  );
}
