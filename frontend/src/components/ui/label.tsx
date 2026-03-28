import { LabelHTMLAttributes } from "react";

type LabelProps = LabelHTMLAttributes<HTMLLabelElement>;

function joinClasses(...parts: Array<string | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export function Label({ className, ...props }: LabelProps) {
  return (
    <label
      className={joinClasses("block text-sm font-medium text-zinc-200", className)}
      {...props}
    />
  );
}
