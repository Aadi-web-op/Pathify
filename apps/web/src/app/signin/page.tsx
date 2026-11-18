import type { Metadata } from "next";
import Link from "next/link";
import { Suspense } from "react";

import { AuthLayout } from "@/components/app/auth-layout";

import { SignInForm } from "./sign-in-form";

export const metadata: Metadata = { title: "Sign in" };

export default function SignInPage() {
  return (
    <AuthLayout
      title="Sign in"
      intro={
        <>
          New to Pathify?{" "}
          <Link href="/signup" className="font-semibold text-sign underline underline-offset-4">
            Create a free account
          </Link>
        </>
      }
    >
      <Suspense>
        <SignInForm />
      </Suspense>
    </AuthLayout>
  );
}
