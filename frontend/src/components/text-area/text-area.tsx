import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import type { ValidationResult } from "@/lib/validation";

interface TextAreaProps {
  onResult?: (result: ValidationResult) => void;
}

function Text_area({ onResult }: TextAreaProps) {
  const [text, setText] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const response = await fetch("http://localhost:8000/jobs-postings/text-base", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ posting_text: text }),
    });
    const data = await response.json();
    console.log("Response from backend:", data);
    
    // Call the onResult callback with the validation result
    if (onResult) {
      onResult({
        classification: data.classification || "ERROR",
        confidence: data.confidence || 0,
      });
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <Textarea placeholder="Enter content..." className="mt-4" value={text} onChange={e => setText(e.target.value)} />
      <Button className="mt-4" type="submit">Check</Button>
    </form>
  );
}

export default Text_area;
