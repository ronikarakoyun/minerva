/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // README.md: cool neutral, very low chroma oklch tokens
        "bg-0":      "oklch(0.16 0.005 240)",
        "bg-1":      "oklch(0.19 0.005 240)",
        "bg-2":      "oklch(0.22 0.006 240)",
        "bg-3":      "oklch(0.26 0.008 240)",
        line:        "oklch(0.30 0.008 240)",
        "line-soft": "oklch(0.25 0.006 240)",
        "fg-0":      "oklch(0.96 0.005 240)",
        "fg-1":      "oklch(0.78 0.008 240)",
        "fg-2":      "oklch(0.58 0.010 240)",
        "fg-3":      "oklch(0.42 0.010 240)",
        accent:      "oklch(0.74 0.045 220)",
        "accent-soft": "oklch(0.30 0.040 220)",
        pos:         "oklch(0.72 0.10 150)",
        neg:         "oklch(0.68 0.13 25)",
        warn:        "oklch(0.78 0.10 80)",
      },
      fontFamily: {
        sans:  ["Inter", "system-ui", "sans-serif"],
        mono:  ['"JetBrains Mono"', "ui-monospace", "monospace"],
        serif: ['"Source Serif 4"', "Georgia", "serif"],
      },
      borderRadius: {
        // 2 (inputs), 3 (cards/panels), 6 (frame)
        DEFAULT: "3px",
        sm:  "2px",
        md:  "3px",
        lg:  "6px",
      },
      spacing: {
        // mock'taki spacing scale: 2, 4, 6, 8, 10, 12, 14, 18, 22, 28
        0.5: "2px",
        1:   "4px",
        1.5: "6px",
        2:   "8px",
        2.5: "10px",
        3:   "12px",
        3.5: "14px",
        4.5: "18px",
        5.5: "22px",
        7:   "28px",
      },
      fontSize: {
        // README.md type scale
        "mono-xs":  ["9.5px",  { lineHeight: "1.4", letterSpacing: "0.05em" }],
        "mono-sm":  ["10px",   { lineHeight: "1.4", letterSpacing: "0.05em" }],
        "mono-md":  ["10.5px", { lineHeight: "1.4", letterSpacing: "0.04em" }],
        "mono-lg":  ["11.5px", { lineHeight: "1.4" }],
        "mono-xl":  ["14px",   { lineHeight: "1.6", letterSpacing: "0.01em" }],
        "stat":     ["22px",   { lineHeight: "1.1", letterSpacing: "-0.015em" }],
        "stat-lg":  ["30px",   { lineHeight: "1.0", letterSpacing: "-0.03em" }],
        "stat-xl":  ["36px",   { lineHeight: "1.0", letterSpacing: "-0.03em" }],
        "h1":       ["28px",   { lineHeight: "1.1", letterSpacing: "-0.014em" }],
        "h1-lg":    ["40px",   { lineHeight: "1.1", letterSpacing: "-0.014em" }],
      },
    },
  },
  plugins: [],
};
