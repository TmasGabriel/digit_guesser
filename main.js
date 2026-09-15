import { canvas, canvas2, clearCanvas, getPixelValues, displayScaledArr, setOnCanvasChangeListener, ctx } from './canvas.js';
import { initNet, fprop, addSoftmax, bprop } from './nn.js';
import { loadMNIST, make2d, poolArr, make1d } from './dataUtils.js';

const iterateButton = document.getElementById("iterateButton");
const clearButton = document.getElementById("clearButton");
const imagesInput = document.getElementById("imagesFile");
const labelsInput = document.getElementById("labelsFile");

let net = initNet([784, 32, 32, 32, 10]);
let learningRate = .005;
let isFrameScheduled = false;

function updateRealTimePrediction() {
    const pixelValues = getPixelValues(canvas);
    const pixelValues2d = make2d(pixelValues, 448);
    let pooledData = pixelValues2d;
    
    for (let i = 0; i < 4; i++) {
        pooledData = poolArr(pooledData);
    }
    
    const image = make1d(pooledData);
    
    displayScaledArr(make2d(image, 28), canvas2);

    const rawOut = fprop(image, net);
    const fpOut = addSoftmax(rawOut);
    const probabilities = fpOut[1][net[1].length - 1];

    updateProbabilityUI(probabilities);
    isFrameScheduled = false;
}

// Scheduled wrapper to throttle execution with requestAnimationFrame
function triggerRealTimeUpdate() {
    if (!isFrameScheduled) {
        isFrameScheduled = true;
        requestAnimationFrame(updateRealTimePrediction);
    }
}

// Updates HTML progress bars
function updateProbabilityUI(probabilities) {
    let maxProb = -1;
    let maxIndex = -1;

    for (let i = 0; i < 10; i++) {
        if (probabilities[i] > maxProb) {
            maxProb = probabilities[i];
            maxIndex = i;
        }
    }

    for (let i = 0; i < 10; i++) {
        const barFill = document.getElementById(`bar-${i}`);
        const valText = document.getElementById(`val-${i}`);
        const percentage = (probabilities[i] * 100).toFixed(1);

        if (barFill && valText) {
            barFill.style.width = `${percentage}%`;
            valText.textContent = `${percentage}%`;

            if (i === maxIndex && maxProb > 0) {
                barFill.classList.add('top-prediction');
            } else {
                barFill.classList.remove('top-prediction');
            }
        }
    }
}

// Listen for draw/erase
setOnCanvasChangeListener(triggerRealTimeUpdate);

clearButton.addEventListener("click", () => {
    clearCanvas();
});

iterateButton.addEventListener("click", async () => {

    const dataset = await loadMNIST(imagesInput, labelsInput);
    let val = 0;

    for (let i = 0; i < dataset.length; i++) {
        for (let j = 0; j < dataset[i].image.length; j++) {
            dataset[i].image[j] = dataset[i].image[j] < 127 ? 0 : 1;
        }
    }

    console.log("done pooling");
    for (let i = 0; i < dataset.length; i++) {
        const rawOut = fprop(dataset[i].image, net);
        const fpOut = addSoftmax(rawOut);

        val += fpOut[1][net[1].length-1][dataset[i].label];
        if (i > 0 && i / 1000 == Math.floor(i / 1000)) {
            console.log(Math.round((val / 1000) * 100000) / 1000);
            val = 0;
            if (learningRate > .001) {
                learningRate -= .0005;
            }
        }
        bprop(dataset[i].label, fpOut[0], fpOut[1], net, dataset[i].image, learningRate);
    }

    triggerRealTimeUpdate();
});
triggerRealTimeUpdate();

document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        ctx.getImageData(0, 0, 1, 1);
    }
});