import type { Metadata } from "next";
import "./globals.css";
import PageNav from "@/components/PageNav";

export const metadata: Metadata = {
  title: "ESP32 Health Monitor — Model Explainer",
  description:
    "Defense-prep walkthrough of the ESP32 TinyML health monitoring project's model results — Category 1, Category 2, Comparison, and Model Selection.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <PageNav />
        {children}
      </body>
    </html>
  );
}
