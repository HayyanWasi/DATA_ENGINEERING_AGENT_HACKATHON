import { ButtonHTMLAttributes, forwardRef } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "ghost";
};

function joinClasses(...parts: Array<string | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "default", type = "button", disabled, onClick, ...props }, ref) => {
    const base =
      "inline-flex items-center justify-center gap-2 rounded-md text-sm transition-all pointer-events-auto disabled:pointer-events-none disabled:opacity-50";
    const variants = {
      default: "bg-zinc-900 text-white hover:bg-zinc-800",
      ghost: "bg-transparent text-zinc-300 hover:bg-zinc-800/60",
    };

    return (
      <button
        ref={ref}
        type={type}
        disabled={disabled}
        onClick={onClick}
        className={joinClasses(base, variants[variant], className)}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";
