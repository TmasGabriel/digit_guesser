import { canvas, canvas2, clearCanvas, getPixelValues, displayScaledArr, setOnCanvasChangeListener, ctx } from './canvas.js';
import { initNet, fprop, bprop, updateWeights } from './nn.js';
import { loadMNIST, make2d, poolArr, make1d, rotateImage, scaleImage, offsetImage } from './dataUtils.js';

const iterateButton = document.getElementById("iterateButton");
const clearButton = document.getElementById("clearButton");
const testButton = document.getElementById("testButton");
const imagesInput = document.getElementById("imagesFile");
const labelsInput = document.getElementById("labelsFile");

const imagesTestInput = document.getElementById("imagesTestFile");
const labelsTestInput = document.getElementById("labelsTestFile");

let net = initNet([784, 512, 512, 512, 10]);
let learningRate = .035;
let dropoutRate = 0.35;
const epochs = 2;
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

    const rawOut = fprop(image, net, 0);
    const probabilities = rawOut[1][net[1].length - 1];

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
    const testingDataset = await loadMNIST(imagesTestInput, labelsTestInput);
    
    

    for (let i = 0; i < testingDataset.length; i++) {
        for (let j = 0; j < testingDataset[i].image.length; j++) {
            testingDataset[i].image[j] = testingDataset[i].image[j] < 127 ? 0 : 1;
        }
    }

    for (let i = 0; i < dataset.length; i++) {
        for (let j = 0; j < dataset[i].image.length; j++) {
            dataset[i].image[j] = dataset[i].image[j] < 127 ? 0 : 1;
        }
    }

    console.log("done pooling");

    let training = [];
    let validation = [];
    for (let e = 0; e < epochs; e++) {
        for (let i = 0; i < dataset.length; i++) {
            const randRot = (Math.random() * 2 - 1) / 8;
            dataset[i].image = rotateImage(dataset[i].image, randRot);
            const randScale = (Math.random() / 4) + .75;
            dataset[i].image = scaleImage(dataset[i].image, randScale);
            const randOffX = Math.floor((Math.random() * 5) - 2);
            const randOffY = Math.floor((Math.random() * 5) - 2);
            dataset[i].image = offsetImage(dataset[i].image, randOffX, randOffY);
        }

        let val = 0;
        let validVal = 0;
        let accuracy = 0;
        let totalAcc = 0;
        let ticker = 0;
        for (let i = 0; i < dataset.length; i++) {

            const rawOut = fprop(dataset[i].image, net, dropoutRate);
            accuracy += rawOut[1][net[1].length-1][dataset[i].label];
            totalAcc += rawOut[1][net[1].length-1][dataset[i].label];

            let sum = 0;
            for (let j = 0; j < 10; j++) {
                const num = dataset[i].label == j ? 1 : 0;
                sum += (rawOut[1][net[1].length-1][j] - num)**2
            }
            const loss = sum / 10;
            val += loss;
            
            const validrawOut = fprop(testingDataset[i - (ticker * 1000)].image, net, 0);
            //validVal += validfpOut[1][net[1].length-1][testingDataset[i - (ticker * 1000)].label];

            let sum2 = 0;
            for (let j = 0; j < 10; j++) {
                const num = testingDataset[i - (ticker * 1000)].label == j ? 1 : 0;
                sum2 += (validrawOut[1][net[1].length-1][j] - num)**2
            }
            const loss2 = sum2 / 10;
            validVal += loss2;
            
            if ((i + 1) / 1000 == Math.floor((i + 1) / 1000)) {
                console.log(Math.round(accuracy * 100) / 100000);

                training.push(`(${Math.floor((i + 1) / 1000)}, ${val})`);
                validation.push(`(${Math.floor((i + 1) / 1000)}, ${validVal})`);

                val = 0;
                validVal = 0;
                accuracy = 0;
                ticker++;

                if (learningRate > .0005) {
                    learningRate -= .0005;
                }
                
            }
            const gradient = bprop(dataset[i].label, rawOut[0], rawOut[1], net, dataset[i].image, rawOut[2], learningRate);
            updateWeights(net, gradient, learningRate);
        }

        console.log(`Epoch accuracy ${e+1}: ${totalAcc / dataset.length}`);

    }

    console.log(training);
    console.log(validation);
    triggerRealTimeUpdate();
});

testButton.addEventListener("click", async () => {
    const dataset = await loadMNIST(imagesTestInput, labelsTestInput);
    let val = 0;

    for (let i = 0; i < dataset.length; i++) {
        for (let j = 0; j < dataset[i].image.length; j++) {
            dataset[i].image[j] = dataset[i].image[j] < 127 ? 0 : 1;
        }
        const randRot = (Math.random() * 2 - 1) / 8;
        dataset[i].image = rotateImage(dataset[i].image, randRot);
        const randScale = (Math.random() / 2) + .5;
        dataset[i].image = scaleImage(dataset[i].image, randScale);
        const randOffX = Math.floor((Math.random() * 5) - 2);
        const randOffY = Math.floor((Math.random() * 5) - 2);
        dataset[i].image = offsetImage(dataset[i].image, randOffX, randOffY);
    }

    for (let i = 0; i < dataset.length; i++) {
        const rawOut = fprop(dataset[i].image, net, 0);

        // console.log(dataset[i].image, dataset[i].label);
        displayScaledArr(make2d(dataset[i].image, 28), canvas2);
        
        await new Promise(resolve => setTimeout(resolve, 2));

        val += rawOut[1][net[1].length-1][dataset[i].label];
        if ((i + 1) / 1000 == Math.floor((i + 1) / 1000)) {
            console.log(Math.round(val * 100) / 100000);
            val = 0;
        }
    }
});
triggerRealTimeUpdate();

document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        ctx.getImageData(0, 0, 1, 1);
    }
});