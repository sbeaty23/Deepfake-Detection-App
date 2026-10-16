import { useRef, useState, type ChangeEvent, type DragEvent } from "react";

function isImage(file: File | undefined): file is File {
  return Boolean(file?.type.startsWith("image/"));
}

function App() {
  const inputRef = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [fileName, setFileName] = useState<string>();
  const [isDragging, setIsDragging] = useState(false);

  function selectFile(file: File | undefined): void {
    if (!isImage(file)) return;
    setFileName(file.name);
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>): void {
    selectFile(event.target.files?.[0]);
  }

  function handleDragEnter(event: DragEvent<HTMLLabelElement>): void {
    event.preventDefault();
    dragDepth.current += 1;
    setIsDragging(true);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>): void {
    event.preventDefault();
  }

  function handleDragLeave(event: DragEvent<HTMLLabelElement>): void {
    event.preventDefault();
    dragDepth.current -= 1;

    if (dragDepth.current === 0) {
      setIsDragging(false);
    }
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>): void {
    event.preventDefault();
    dragDepth.current = 0;
    setIsDragging(false);

    const file = event.dataTransfer.files[0];
    if (!isImage(file)) return;

    const transfer = new DataTransfer();
    transfer.items.add(file);

    if (inputRef.current) {
      inputRef.current.files = transfer.files;
    }

    selectFile(file);
  }

  return (
    <main className="page-shell">
      <section className="upload-card" aria-labelledby="upload-title">
        <div className="upload-heading">
          <span className="eyebrow">Image upload</span>
          <h1 id="upload-title">Choose an image</h1>
          <p>Drag and drop a file here, or browse from your device.</p>
        </div>

        <label
          className={`drop-zone${isDragging ? " is-dragging" : ""}`}
          htmlFor="image-input"
          onDragEnter={handleDragEnter}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            ref={inputRef}
            id="image-input"
            type="file"
            accept="image/*"
            onChange={handleChange}
          />

          <span className="upload-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5" />
              <path d="M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
            </svg>
          </span>

          <span className="drop-copy">
            <strong>{fileName ?? "Drop your image here"}</strong>
            <span>PNG, JPG, or WEBP</span>
          </span>

          <span className="browse-button">Browse files</span>
        </label>
      </section>
    </main>
  );
}

export default App;
