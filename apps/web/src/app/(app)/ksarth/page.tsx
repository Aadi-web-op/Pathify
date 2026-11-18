import { Suspense } from "react";

import { KsarthChat } from "./ksarth-chat";

export default function KsarthPage() {
  return (
    <Suspense>
      <KsarthChat />
    </Suspense>
  );
}
