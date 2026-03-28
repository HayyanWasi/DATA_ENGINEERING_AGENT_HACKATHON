import { HTMLAttributes, ThHTMLAttributes, TdHTMLAttributes } from "react";

type Classy<T> = T & { className?: string };

function joinClasses(...parts: Array<string | undefined>) {
  return parts.filter(Boolean).join(" ");
}

export function Table({ className, ...props }: Classy<HTMLAttributes<HTMLTableElement>>) {
  return <table className={joinClasses("w-full caption-bottom text-sm", className)} {...props} />;
}

export function TableHeader({ className, ...props }: Classy<HTMLAttributes<HTMLTableSectionElement>>) {
  return <thead className={joinClasses("[&_tr]:border-b", className)} {...props} />;
}

export function TableBody({ className, ...props }: Classy<HTMLAttributes<HTMLTableSectionElement>>) {
  return <tbody className={joinClasses("[&_tr:last-child]:border-0", className)} {...props} />;
}

export function TableRow({ className, ...props }: Classy<HTMLAttributes<HTMLTableRowElement>>) {
  return (
    <tr
      className={joinClasses("border-b border-zinc-800 transition-colors", className)}
      {...props}
    />
  );
}

export function TableHead({ className, ...props }: Classy<ThHTMLAttributes<HTMLTableCellElement>>) {
  return (
    <th
      className={joinClasses("h-12 px-4 text-left align-middle font-medium", className)}
      {...props}
    />
  );
}

export function TableCell({ className, ...props }: Classy<TdHTMLAttributes<HTMLTableCellElement>>) {
  return <td className={joinClasses("p-4 align-middle", className)} {...props} />;
}
