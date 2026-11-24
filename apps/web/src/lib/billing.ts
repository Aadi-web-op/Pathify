import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, ApiError, unwrap } from "./api";
import type { components } from "./api-schema";
import { ME_KEY } from "./session";

export type Billing = components["schemas"]["BillingOut"];
type Order = components["schemas"]["OrderOut"];

export function useBilling() {
  return useQuery<Billing, ApiError>({
    queryKey: ["billing"],
    queryFn: () => unwrap(api.GET("/api/billing")),
  });
}

export const rupees = (paise: number) =>
  `₹${(paise / 100).toLocaleString("en-IN")}`;

type RazorpayResponse = {
  razorpay_order_id: string;
  razorpay_payment_id: string;
  razorpay_signature: string;
};
type RazorpayInstance = {
  open: () => void;
  on: (
    event: string,
    handler: (r: { error?: { description?: string } }) => void,
  ) => void;
};
declare global {
  interface Window {
    Razorpay?: new (options: Record<string, unknown>) => RazorpayInstance;
  }
}

/** Loads Razorpay Checkout only when someone actually starts a payment. */
function loadCheckout(): Promise<void> {
  if (window.Razorpay) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://checkout.razorpay.com/v1/checkout.js";
    script.onload = () => resolve();
    script.onerror = () =>
      reject(
        new Error("Couldn't load the payment window. Check your connection."),
      );
    document.body.appendChild(script);
  });
}

/** Create the order on the server, open Checkout, then have the server verify the signature. */
export function useBuy() {
  const queryClient = useQueryClient();
  return useMutation<Billing, Error, "family_plus" | "school">({
    mutationFn: async (product) => {
      const order: Order = await unwrap(
        api.POST("/api/billing/orders", { body: { product } }),
      );
      await loadCheckout();
      const paid = await new Promise<RazorpayResponse>((resolve, reject) => {
        const checkout = new window.Razorpay!({
          key: order.key_id,
          order_id: order.order_id,
          amount: order.amount,
          currency: order.currency,
          name: "Pathify",
          description: order.description,
          prefill: { name: order.name, email: order.email },
          theme: { color: "#4b3fd8" },
          handler: resolve,
          modal: { ondismiss: () => reject(new Error("Payment cancelled.")) },
        });
        checkout.on("payment.failed", (r) =>
          reject(
            new Error(r.error?.description ?? "The payment didn't go through."),
          ),
        );
        checkout.open();
      });
      return unwrap(
        api.POST("/api/billing/verify", {
          body: {
            order_id: paid.razorpay_order_id,
            payment_id: paid.razorpay_payment_id,
            signature: paid.razorpay_signature,
          },
        }),
      );
    },
    onSuccess: (billing) => {
      queryClient.setQueryData(["billing"], billing);
      void queryClient.invalidateQueries({ queryKey: ME_KEY });
    },
  });
}
