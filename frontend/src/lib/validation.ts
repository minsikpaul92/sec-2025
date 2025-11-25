export type ValidationResult = {
  classification: string;
  confidence: number;
  missing_fields?: string[];
  transparency_score?: number;
};
