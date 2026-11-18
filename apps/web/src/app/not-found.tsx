import { Logo } from "@/components/brand/logo";
import { ButtonLink } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="gutter grid min-h-dvh grid-rows-[auto_1fr]">
      <header className="mx-auto flex h-18 w-full max-w-5xl items-center">
        <Logo />
      </header>
      <main id="main" className="mx-auto grid w-full max-w-5xl content-center gap-5 pb-24">
        <p className="font-mono text-muted">404</p>
        <h1 className="max-w-xl text-[2.4rem] font-extrabold leading-tight">This road doesn&apos;t go anywhere.</h1>
        <p className="text-muted">The page may have moved, or the link has a typo.</p>
        <ButtonLink href="/" className="justify-self-start">
          Back to Pathify
        </ButtonLink>
      </main>
    </div>
  );
}
