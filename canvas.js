export const canvas = document.getElementById("myCanvas");

// { willReadFrequently: true } // in case there are tab switch delays
export const ctx = canvas.getContext("2d", { willReadFrequently: true });

export const canvas2 = document.getElementById("theCanvas");
export const ctx2 = canvas2.getContext("2d", { willReadFrequently: true });

let isDrawing = false;
let isErasing = false;
let lastX = 0;
let lastY = 0;

const brushRadius = 10;
const bufferDistance = 30; // For mouse outside canvas

let onCanvasChangeCallback = null;

// Drawing/erasing listner
export function setOnCanvasChangeListener(callback) {
    onCanvasChangeCallback = callback;
}

function notifyCanvasChange() {
    if (typeof onCanvasChangeCallback === 'function') {
        onCanvasChangeCallback();
    }
}

clearCanvas();

// Prevent context menu pop up on right click
canvas.addEventListener('contextmenu', function(event) {
    event.preventDefault(); 
});

// Canvas coordinate mapping
export function getCanvasCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    return { x, y, rect };
}

function drawLine(x, y, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = brushRadius * 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    lastX = x;
    lastY = y;
}

// Pointer Down Handler
canvas.addEventListener('pointerdown', (e) => {
    if (e.button === 0) {
        isDrawing = true;
        isErasing = false;
    } else if (e.button === 2) {
        isErasing = true;
        isDrawing = false;
    } else {
        return;
    }

    canvas.setPointerCapture(e.pointerId);

    const { x, y } = getCanvasCoordinates(e);
    lastX = x;
    lastY = y;

    ctx.beginPath();
    ctx.fillStyle = isErasing ? "white" : "black";
    ctx.arc(x, y, brushRadius, 0, Math.PI * 2);
    ctx.fill();

    notifyCanvasChange();
});

// Pointer Move Handler
canvas.addEventListener('pointermove', (e) => {
    if (!isDrawing && !isErasing) return;

    const { x, y, rect } = getCanvasCoordinates(e);

    const isWithinBuffer = 
        e.clientX >= rect.left - bufferDistance &&
        e.clientX <= rect.right + bufferDistance &&
        e.clientY >= rect.top - bufferDistance &&
        e.clientY <= rect.bottom + bufferDistance;

    if (isWithinBuffer) {
        const color = isErasing ? "white" : "black";
        drawLine(x, y, color);
        notifyCanvasChange();
    } else {
        stopDrawing(e);
    }
});

function stopDrawing(e) {
    if (isDrawing || isErasing) {
        isDrawing = false;
        isErasing = false;
        if (e.pointerId && canvas.hasPointerCapture(e.pointerId)) {
            canvas.releasePointerCapture(e.pointerId);
        }
    }
}

canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointercancel', stopDrawing);

export function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    notifyCanvasChange();
}

export function getBrightness(r, g, b) {
    return r * 0.299 + g * 0.587 + b * 0.114;
}

export function getPixelValues(targetCanvas) {
    const width = targetCanvas.width;
    const height = targetCanvas.height;
    let pixelValueArr = [];

    const canvasCtx = targetCanvas.getContext("2d", { willReadFrequently: true });
    const canvasImage = canvasCtx.getImageData(0, 0, width, height);
    const canvasData = canvasImage.data;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const canvasIndex = (y * width + x) * 4;

            const r = canvasData[canvasIndex];
            const g = canvasData[canvasIndex + 1];
            const b = canvasData[canvasIndex + 2];
            const brightness = getBrightness(r, g, b);

            if (brightness > 127) {
                pixelValueArr.push(0);
            }
            else {
                pixelValueArr.push(1);
            }
        }
    }
    return pixelValueArr;
}

export function displayScaledArr(arr2d, targetCanvas) {
    const targetCtx = targetCanvas.getContext("2d", { willReadFrequently: true });
    const rows = arr2d.length;
    const cols = arr2d[0].length;
    
    // Calculate pixel block size
    const scaleX = targetCanvas.width / cols;
    const scaleY = targetCanvas.height / rows;

    for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
            targetCtx.fillStyle = arr2d[y][x] === 1 ? 'black' : 'white';
            targetCtx.fillRect(x * scaleX, y * scaleY, scaleX, scaleY);
        }
    }
}