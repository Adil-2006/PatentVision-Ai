"use client";

import { useEffect, useState } from "react";
import { fetchBlob } from "@/lib/api";

interface Props {
  /** API path such as /ai/results/3/download?kind=mask. Null renders the placeholder. */
  src: string | null;
  alt: string;
  className?: string;
  placeholder?: string;
  /** Change this to force a reload (e.g. after reprocessing). */
  version?: string | number;
}

/** <img> for protected API files: fetches with the bearer token and shows a blob URL. */
export function SecureImage({ src, alt, className = "", placeholder = "Not available", version }: Props) {
  const [url, setUrl] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    let revoked = false;
    let objectUrl: string | null = null;
    setUrl(null);
    setFailed(false);
    if (!src) return;

    let attempts = 0;
    const fetchWithRetry = () => {
      fetchBlob(src)
        .then((blob) => {
          if (revoked) return;
          objectUrl = URL.createObjectURL(blob);
          setUrl(objectUrl);
        })
        .catch(() => {
          if (revoked) return;
          if (attempts < 3) {
            attempts++;
            setTimeout(fetchWithRetry, 800 * attempts);
          } else {
            setFailed(true);
          }
        });
    };

    fetchWithRetry();

    return () => {
      revoked = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src, version, retryKey]);

  if (!src || failed) {
    return (
      <div className={`flex flex-col items-center justify-center gap-1.5 text-xs text-slate-400 ${className}`}>
        <span>{failed ? "Could not load" : placeholder}</span>
        {failed && (
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setRetryKey((k) => k + 1);
            }}
            className="rounded border border-slate-300 bg-white px-2 py-0.5 text-[10px] font-medium text-blue-600 shadow-sm hover:bg-slate-50"
          >
            Retry
          </button>
        )}
      </div>
    );
  }
  if (!url) return <div className={`animate-pulse bg-slate-100 ${className}`} aria-label="Loading" />;
  // eslint-disable-next-line @next/next/no-img-element
  return <img src={url} alt={alt} className={className} />;
}
