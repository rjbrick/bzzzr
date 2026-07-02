// Buzzzr config — set this ONCE, then never overwrite it on future deploys.
// The anon/public key is safe to expose in a static site (it only allows the
// access your Row Level Security policies permit).
//
// Paste your anon/public key on the SUPABASE_ANON line.
// GADS_ID / GADS_LABEL are OPTIONAL — fill them in once you create a Google Ads
// conversion action (leave as-is to keep ad tracking off).

window.BUZZZR_CONFIG = {
  SUPABASE_URL:  "https://tdqzpuojiidorqnrgmdw.supabase.co",
  SUPABASE_ANON: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRkcXpwdW9qaWlkb3JxbnJnbWR3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODI4MjA5MTEsImV4cCI6MjA5ODM5NjkxMX0.cGWuK88Dt_QVcuQbYR2B563KZ9ULYOFxTKFE-dVDzzI",

  GADS_ID:    "AW-18293389567",   // e.g. "AW-1234567890"
  GADS_LABEL: "KXI2CNDcmskcEP_x-5JE"    // e.g. "AbC-D_efGhIjKlMnOp"
};
