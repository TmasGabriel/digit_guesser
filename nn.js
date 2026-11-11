function initBiases(numNodes) {
    return new Array(numNodes).fill(0); // Biases should start at 0
}

function initWeights(numNodes, numPrevNodes) {
    if (numPrevNodes == null || numPrevNodes <= 0) {
        return -1;
    } 

    // Pre-allocation apperntly slower than dynamic
    const weights = []; 
    const limit = Math.sqrt(6 / numPrevNodes); // <-- Math.sqrt(2 / numPrevNodes) = He / Kaiming Initialization
    for (let i = 0; i < numNodes; i++) {
        weights.push([]);
        for (let j = 0; j < numPrevNodes; j++) {
            weights[i].push((Math.random() * 2 - 1) * limit);
        }
    }
    return weights;
}

export function initNet(numNodesArr) {
    const biases = [];
    const weights = [];
    for (let i = 1; i < numNodesArr.length; i++) {
        biases.push(initBiases(numNodesArr[i]));
        weights.push(initWeights(numNodesArr[i], numNodesArr[i - 1]));
    }
    return [biases, weights];
}

export function softmax(rawOut) {
    const max = Math.max(...rawOut);
    const exps = rawOut.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);

    return exps.map(x => x / sum);
}

const leakyReLUHP = 0.01;

export function fprop(initInput, net, dropoutRate) {
    let input = initInput;
    const lenNet = net[0].length;
    const logits = [];
    const activations = [];
    const dropoutMasks = [];
    
    for (let i = 0; i < lenNet; i++) {
        const scale = i == lenNet - 1 ? 1 : 1 / (1 - dropoutRate);
        const mask = net[0][i].map(() => (i == lenNet - 1 || Math.random() > dropoutRate ? 1 : 0));
        const layerActivations = [];
        const layerLogits = [];
        for (let j = 0; j < net[0][i].length; j++) {
            let f = 0;
            for (let k = 0; k < input.length; k++) {
                f += input[k] * net[1][i][j][k]; // mult by weight
            }
            f += net[0][i][j]; // add bias
            f *= mask[j] * scale; // dropout
            layerLogits.push(f);
            if (i+1 < lenNet) {
                const leakyReLU = f > 0 ? f : leakyReLUHP * f; // <- ReLU for non output layers
                layerActivations.push(leakyReLU);
            }
            
        }

        dropoutMasks.push(mask);
        logits.push(layerLogits);
        activations.push(i+1 == lenNet ? softmax(layerLogits) : layerActivations);
        input = layerActivations;
    }
    return [logits, activations, dropoutMasks];
}

// for back prop of weights: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU] * (previous node activation))
// for back prop of biases: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU])
// for back prop of previous node: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU] * (weight))

export function bprop(realVal, rawOut, aFuncOut, net, initInput, dropoutMasks, learningRate) {
    const lastLayer = net[1].length - 1;

    const biasGradients = new Array(net[1].length);
    const weightGradients = new Array(net[1].length);

    let outputGradient = [];

    for (let i = lastLayer; i >= 0; i--) { // For each layer in the network staring with the last layer
        const prevActivation = (i == 0) ? initInput : aFuncOut[i - 1];
        const numNodes = net[1][i].length;
        const numInputs = prevActivation.length;

        const delta = new Array(numNodes);
        biasGradients[i] = new Array(net[1][i].length);
        weightGradients[i] = new Array(net[1][i].length);

        // Calc dL/dz
        if (i == lastLayer) { // On the last layer calc cross entropy output softmax
            for (let j = 0; j < numNodes; j++) {
                const target = (realVal == j) ? 1 : 0;
                delta[j] = aFuncOut[i][j] - target;
            }
        } else {
            for (let j = 0; j < numNodes; j++) { // For hidden layers use ReLU unless node is dropped
                const isDropped = dropoutMasks[i][j] == 0;
                const ReLUdx = rawOut[i][j] > 0 ? 1 : leakyReLUHP;

                delta[j] = isDropped ? 0 : outputGradient[j] * ReLUdx;
            }
        }

        // Compute output gradient of next node
        const prevGrad = new Array(numInputs).fill(0);
        for (let j = 0; j < numNodes; j++) {
            if (delta[j] == 0) continue; // Skip dropped or dead nodes

            for (let k = 0; k < numInputs; k++) {
                prevGrad[k] += delta[j] * net[1][i][j][k];
            }
        }

        // Update weights
        for (let j = 0; j < numNodes; j++) {
            if (delta[j] == 0) { // Skip dropped or dead nodes
                biasGradients[i][j] = 0;
                weightGradients[i][j] = new Array(net[1][i][j].length).fill(0);
                continue;
            }

            biasGradients[i][j] = delta[j]; // Update bias

            weightGradients[i][j] = new Array(net[1][i][j].length);
            for (let k = 0; k < numInputs; k ++) {
                weightGradients[i][j][k] = delta[j] * prevActivation[k];  // Update weight
            }
        }

        outputGradient = prevGrad;
    }
    return { biases: biasGradients, weights: weightGradients };
}



export function updateWeights(net, gradients, learningRate) {
    for (let i = 0; i < net[1].length; i++) {
        for (let j = 0; j < net[1][i].length; j++) {

            net[0][i][j] -= learningRate * gradients.biases[i][j];

            // Clip and update weights
            for (let k = 0; k < net[1][i][j].length; k++) {
                net[1][i][j][k] -= learningRate * gradients.weights[i][j][k];
            }
        }
    }
}