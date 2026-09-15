export const POOL_SIZE = 64;
export const THRESHOLD = 128;

let pooledContainer = null;

export function poolCanvas(sourceCanvas, poolSize) {
    const sourceCtx = sourceCanvas.getContext("2d");

    const sourceWidth = sourceCanvas.width;
    const sourceHeight = sourceCanvas.height;

    const newWidth = Math.max(1, Math.floor(sourceWidth / poolSize));
    const newHeight = Math.max(1, Math.floor(sourceHeight / poolSize));

    const sourceImage = sourceCtx.getImageData(0, 0, sourceWidth, sourceHeight);
    const sourceData = sourceImage.data;

    const outputCanvas = document.createElement("canvas");
    outputCanvas.width = newWidth;
    outputCanvas.height = newHeight;

    const outputCtx = outputCanvas.getContext("2d");

    const outputImage = outputCtx.createImageData(
        newWidth,
        newHeight
    );

    const outputData = outputImage.data;

    for (let y = 0; y < newHeight; y++) {
        for (let x = 0; x < newWidth; x++) {

            let hasBlackPixel = false;

            for (let offsetY = 0; offsetY < poolSize; offsetY++) {
                for (let offsetX = 0; offsetX < poolSize; offsetX++) {

                    const sourceX = x * poolSize + offsetX;
                    const sourceY = y * poolSize + offsetY;

                    if (
                        sourceX >= sourceWidth ||
                        sourceY >= sourceHeight
                    ) {
                        continue;
                    }

                    const sourceIndex =
                        (sourceY * sourceWidth + sourceX) * 4;

                    const r = sourceData[sourceIndex];
                    const g = sourceData[sourceIndex + 1];
                    const b = sourceData[sourceIndex + 2];

                    // Since we're only using black and white,
                    // we only need to determine whether
                    // this pixel is black.
                    const brightness =
                        (r + g + b) / 3;

                    if (brightness < THRESHOLD) {
                        hasBlackPixel = true;
                        break;
                    }
                }

                if (hasBlackPixel) {
                    break;
                }
            }

            // Black if the region contained ANY black pixel.
            // Otherwise white.
            const value = hasBlackPixel ? 0 : 255;

            const outputIndex =
                (y * newWidth + x) * 4;

            outputData[outputIndex] = value;
            outputData[outputIndex + 1] = value;
            outputData[outputIndex + 2] = value;
            outputData[outputIndex + 3] = 255;
        }
    }

    outputCtx.putImageData(outputImage, 0, 0);

    return outputCanvas;
}

export function showPooledCanvas(pooledCanvas, canvasContainer) {
    if (pooledContainer) {
        pooledContainer.remove();
    }

    pooledContainer = document.createElement("div");
    pooledContainer.classList.add("canvas-container");

    pooledCanvas.classList.add("pooled-canvas");

    const label = document.createElement("div");
    label.classList.add("canvas-label");

    label.textContent = `${pooledCanvas.width} x ${pooledCanvas.height}`;

    pooledContainer.appendChild(pooledCanvas);
    pooledContainer.appendChild(label);

    canvasContainer.appendChild(pooledContainer);
}

export async function loadImages(file) {
    const buffer = await file.arrayBuffer();
    const view = new DataView(buffer);

    const magic = view.getUint32(0, false);
    const numImages = view.getUint32(4, false);
    const rows = view.getUint32(8, false);
    const cols = view.getUint32(12, false);

    if (magic !== 2051) {
        throw new Error("Not a valid MNIST image file");
    }

    const imageSize = rows * cols;
    const images = [];

    for (let i = 0; i < numImages; i++) {
        const start = 16 + i * imageSize;
        const image = new Uint8Array(buffer, start, imageSize);
        images.push(image);
    }

    return {numImages, rows, cols, images};
}

export async function loadLabels(file) {
    const buffer = await file.arrayBuffer();
    const view = new DataView(buffer);

    const magic = view.getUint32(0, false);
    const numLabels = view.getUint32(4, false);

    if (magic !== 2049) {
        throw new Error("Not a valid MNIST label file");
    }

    const labels = new Uint8Array(buffer, 8, numLabels);

    return {numLabels, labels};
}

export async function loadMNIST(imagesInput, labelsInput) {
    const imageFile = imagesInput.files[0];
    const labelFile = labelsInput.files[0];

    const imageData = await loadImages(imageFile);
    const labelData = await loadLabels(labelFile);

    if (imageData.numImages !== labelData.numLabels) {
        throw new Error("Images and labels don't have the same count");
    }

    const dataset = [];

    for (let i = 0; i < imageData.numImages; i++) {
        dataset.push({
            image: imageData.images[i],
            label: labelData.labels[i]
        });
    }

    return dataset;
}

export function make2d(arr, size) {
    let x = 0;
    let y = 0;
    let arr2d = []; 
    arr2d.push([]);
    for (let i = 0; i < arr.length; i++) {
        if (i / size == Math.floor(i / size) && i != 0) {
            arr2d.push([]);
            x = 0;
            y++;
        }
        arr2d[y].push(arr[i]);
        x++;
    }
    return arr2d;
}

export function make1d(arr) {
    let arr1d = [];
    for (let i = 0; i < arr.length; i++) {
        for (let j = 0; j < arr.length; j++) {
            arr1d.push(arr[i][j]);
        }
    }
    return arr1d;
}

export function poolArr(arr) {
    const compressed = [];

    for (let y = 0; y < arr.length; y+=2) {
        compressed.push([]);
        for (let x = 0; x < arr[y].length; x+=2) {
            const maxPixel = Math.max(arr[y][x], arr[y][x+1], arr[y+1][x], arr[y+1][x+1]);
            compressed[y/2][x/2] = maxPixel;
        }
    }
    return compressed;
}