import { useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";

function Text_area() {
  const [text, setText] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const response = await fetch("/api/text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await response.json();
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <Textarea placeholder="Nhập nội dung..." className="mt-4" value={text} onChange={e => setText(e.target.value)} />
      <Button className="mt-4" type="submit">Check</Button>
    </form>
  );
}

export default Text_area;
