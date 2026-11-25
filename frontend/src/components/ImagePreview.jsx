// Shows the input image and the intermediate images returned by the backend.
export default function ImagePreview({ images }) {
  if (!images) return null;

  return (
    <div className="image-row">
      <figure>
        <img src={images.input} alt="Input" />
        <figcaption>Input (resized)</figcaption>
      </figure>
      <figure>
        <img src={images.grayscale} alt="Grayscale" />
        <figcaption>Preprocessed grayscale</figcaption>
      </figure>
    </div>
  );
}
