import { InputHTMLAttributes, forwardRef } from "react";

type InputProps = InputHTMLAttributes<HTMLInputElement>;

function joinClasses(...parts: Array<string | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, ...props }, ref) => {
    return (
      <input
        ref={ref}
        className={joinClasses(
          "w-full rounded-md border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm text-white outline-none ring-0 placeholder:text-zinc-500 focus:border-zinc-500",
          className
        )}
        {...props}
      />
    );
  }
);

Input.displayName = "Input";
