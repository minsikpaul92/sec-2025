import { useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

function Form() {
  // Các state cho form
  const [title, setTitle] = useState("");
  const [salary, setSalary] = useState("");
  const [customSalary, setCustomSalary] = useState("");
  const [location, setLocation] = useState("");
  const [employer, setEmployer] = useState("");
  const [description, setDescription] = useState("");
  const [requirements, setRequirements] = useState("");
  const [benefits, setBenefits] = useState("");
  const [employmentType, setEmploymentType] = useState("");
  const [aiUsed, setAiUsed] = useState("");

  // Xử lý submit
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Chọn đúng lương nếu là "other"
    const salaryValue = salary === "other" ? customSalary : salary;

    // Tổng hợp dữ liệu gửi backend
    const payload = {
      title,
      salary: salaryValue,
      location,
      employer,
      description,
      requirements,
      benefits,
      employmentType,
      aiUsed,
    };

    // Gửi POST request tới API backend (ví dụ /api/job)
    const response = await fetch("/api/job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    // Xử lý phản hồi nếu cần
    const data = await response.json();
    console.log(data);
  };

  return (
    <form onSubmit={handleSubmit}>
      <Input type="text" placeholder="Job title..." className="mt-4" value={title} onChange={e => setTitle(e.target.value)} />
      <select className="mt-4 p-2 border rounded" value={salary} onChange={e => setSalary(e.target.value)}>
        <option value="" disabled>Salary</option>
        <option value="under20k">Under 20k</option>
        <option value="20k-30k">$20,000 - $30,000</option>
        <option value="30k-40k">$30,000 - $40,000</option>
        <option value="40k-50k">$40,000 - $50,000</option>
        <option value="other">Other (specify below)</option>
      </select>
      {salary === "other" && (
        <Input
          type="number"
          className="mt-4"
          placeholder="Enter specific salary"
          value={customSalary}
          onChange={e => setCustomSalary(e.target.value)}
        />
      )}
      <Input type="text" placeholder="Location..." className="mt-4" value={location} onChange={e => setLocation(e.target.value)} />
      <Input type="text" placeholder="Employer..." className="mt-4" value={employer} onChange={e => setEmployer(e.target.value)} />
      <Input type="text" placeholder="Description..." className="mt-4" value={description} onChange={e => setDescription(e.target.value)} />
      <Input type="text" placeholder="Requirements..." className="mt-4" value={requirements} onChange={e => setRequirements(e.target.value)} />
      <Input type="text" placeholder="Benefits..." className="mt-4" value={benefits} onChange={e => setBenefits(e.target.value)} />
      <select className="mt-4 p-2 border rounded" value={employmentType} onChange={e => setEmploymentType(e.target.value)}>
        <option value="" disabled>Employment Type</option>
        <option value="fulltime">Full Time</option>
        <option value="parttime">Part Time</option>
        <option value="intern">Internship</option>
      </select>
      <select className="mt-4 p-2 border rounded" value={aiUsed} onChange={e => setAiUsed(e.target.value)}>
        <option value="" disabled>AI used</option>
        <option value="yes">Yes</option>
        <option value="no">No</option>
      </select>
      <Button className="mt-4" type="submit">Check</Button>
    </form>
  );
}

export default Form;
