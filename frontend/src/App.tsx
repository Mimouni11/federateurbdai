import { useState } from "react";
import { Splash } from "./components/Splash";
import { ToolPicker } from "./components/ToolPicker";
import { Detector } from "./components/Detector";
import { Generator } from "./components/Generator";
import "./App.css";

type View = "splash" | "picker" | "detector" | "generator";

export default function App() {
  const [view, setView] = useState<View>("splash");
  const home = () => setView("picker");

  if (view === "splash")    return <Splash onDone={() => setView("picker")} />;
  if (view === "picker")    return <ToolPicker onPick={setView} />;
  if (view === "detector")  return <Detector onHome={home} />;
  if (view === "generator") return <Generator onHome={home} />;

  return null;
}
