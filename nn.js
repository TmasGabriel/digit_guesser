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

export function addSoftmax(fpropOut) {
    fpropOut[1][fpropOut[1].length - 1] = softmax(fpropOut[1][fpropOut[1].length - 1]);
    return fpropOut;
}

export function fprop(initInput, net, dropoutRate) {
    let input = initInput;
    const lenNet = net[0].length;
    const rawOut = [];
    const actiOut = [];
    const dropoutMasks = [];
    
    for (let k = 0; k < lenNet; k++) {
        const scale = k == lenNet - 1 ? 1 : 1 / (1 - dropoutRate);
        const mask = net[0][k].map(() => (k == lenNet - 1 || Math.random() > dropoutRate ? 1 : 0));
        const actiLayer = [];
        const rawLayer = [];
        for (let i = 0; i < net[0][k].length; i++) {
            let f = 0;
            for (let j = 0; j < input.length; j++) {
                f += input[j] * net[1][k][i][j]; // mult by weight
            }
            f += net[0][k][i]; // add bias
            f *= (mask[i] * scale); // dropout
            const y = k+1 == lenNet ? f : Math.max(0, f); // <- ReLU for non output layers
            actiLayer.push(y);
            rawLayer.push(f);
        }
        dropoutMasks.push(mask);
        rawOut.push(rawLayer);
        actiOut.push(actiLayer);
        input = actiLayer;
    }
    return [rawOut, actiOut, dropoutMasks];
}

// for back prop of weights: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU] * (previous node activation))
// for back prop of biases: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU])
// for back prop of previous node: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU] * (weight))

export function bprop(realVal, rawOut, aFuncOut, net, initInput, dropoutMasks, learningRate, dropoutRate) {
    const lastLayer = net[1].length - 1;
    let gradient = [];

    for (let i = lastLayer; i >= 0; i--) { // For each layer in the network staring with the last layer
        const prevActivation = (i == 0) ? initInput : aFuncOut[i - 1];
        const numNodes = net[1][i].length;
        const numInputs = prevActivation.length;

        const delta = new Array(numNodes);

        // Calc dL/dz
        if (i == lastLayer) { // On the last layer calc cross entropy output softmax
            for (let j = 0; j < numNodes; j++) {
                const target = (realVal == j) ? 1 : 0;
                delta[j] = aFuncOut[i][j] - target;
            }
        } else {
            for (let j = 0; j < numNodes; j++) { // For hidden layers use ReLU unless node is dropped
                const isDropped = dropoutMasks[i][j] == 0;
                const ReLUdx = rawOut[i][j] > 0 ? 1 : 0;

                delta[j] = isDropped ? 0 : gradient[j] * ReLUdx;
            }
        }

        // Compute gradient to send back (must happen before weights are modified)
        const prevGrad = new Array(numInputs).fill(0);
        for (let j = 0; j < numNodes; j++) {
            if (delta[j] == 0) continue; // Skip dropped or dead nodes

            for (let k = 0; k < numInputs; k++) {
                prevGrad[k] += delta[j] * net[1][i][j][k];
            }
        }

        for (let j = 0; j < numNodes; j++) {
            if (delta[j] == 0) continue; // Skip dropped or dead nodes

            net[0][i][j] -= learningRate * delta[j]; // Update bias

            for (let k = 0; k < numInputs; k ++) {
                const weightGrad = delta[j] * prevActivation[k];
                net[1][i][j][k] -= learningRate * weightGrad;
            }
        }

        gradient = prevGrad;
    }
}