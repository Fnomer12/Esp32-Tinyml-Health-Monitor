"use client";


import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "Category 1" },
  { href: "/category2", label: "Category 2" },
  { href: "/category3", label: "Category 3" },
  { href: "/comparison", label: "Comparison" },
  { href: "/model-selection", label: "Model Selection" },
];

export default function PageNav() {
  const pathname = usePathname();

  return (
    <div className="pagenav">
      <div className="pagenav-inner">
        {TABS.map((tab) => {
          const active = pathname === tab.href;
          return (
            <Link
              key={tab.href}
              href={tab.href}
              className={["pagenav-btn", active ? "active" : ""].filter(Boolean).join(" ")}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
    </div>
  );
}