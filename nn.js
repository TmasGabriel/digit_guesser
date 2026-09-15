function initBiases(numNodes) {
    return new Array(numNodes).fill(0); // Biases should start at 0
}

function initWeights(numNodes, numPrevNodes) {
    if (numPrevNodes == null || numPrevNodes <= 0) {
        return -1;
    } 

    // Pre-allocation apperntly slower than dynamic
    const weights = [];
    for (let i = 0; i < numNodes; i++) {
        weights.push([]);
        for (let j = 0; j < numPrevNodes; j++) {
            weights[i].push((Math.random() * 2 - 1) * Math.sqrt(2 / numNodes)); // <-- Math.sqrt(2 / numNodes) = He / Kaiming Initialization
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

export function fprop(initInput, net) {
    let input = initInput;
    const lenNet = net[0].length;
    const rawOut = [];
    const actiOut = [];
    for (let k = 0; k < lenNet; k++) {
        const actiLayer = [];
        const rawLayer = [];
        for (let i = 0; i < net[0][k].length; i++) {
            let f = 0;
            for (let j = 0; j < input.length; j++) {
                f += input[j] * net[1][k][i][j];
            }
            f += net[0][k][i];
            const y = k+1 == lenNet ? f : Math.max(0, f); // <- ReLU for non output layers
            actiLayer.push(y);
            rawLayer.push(f);
        }
        rawOut.push(rawLayer);
        actiOut.push(actiLayer);
        input = actiLayer;
    }
    return [rawOut, actiOut];
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

// for back prop of weights: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU] * (previous node activation))
// for back prop of biases: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU])
// for back prop of previous node: summation of all nodes
// ((predicted - target) * (1 if f > 0, 0 if f < 0)[with f being the activation value BEFORE going through ReLU] * (weight))

export function bprop(realVal, rawOut, aFuncOut, net, initInput, learningRate) {

    let gradient = [];
    for (let i = 0; i < net[1][net[1].length-1].length; i++) {
        gradient[i] = realVal == i ? aFuncOut[aFuncOut.length-1][i] - 1 : aFuncOut[aFuncOut.length-1][i];
    }

    for (let i = net[1].length-1; i >= 0; i--) { // For each layer in the network

        let newGradientArr = [];
        for (let j = 0; j < net[1][i][0].length; j++) { // For each node in that layer
            let newGradient = 0;
            for (let k = 0; k < net[1][i].length; k++) { // For each weight in that node
                newGradient += (gradient[k] * net[1][i][k][j]);
            }
            newGradientArr.push(newGradient);
        }

        const prevActivation = i == 0 ? initInput : aFuncOut[i - 1];
        for (let j = 0; j < net[1][i].length; j++) { // For each node in that layer
            let ReLUdx = 1;
            if (i != net[1].length-1) {
                ReLUdx = rawOut[i][j] > 0 ? 1 : 0;
            }
            const biasGradient = gradient[j] * ReLUdx;
            net[0][i][j] -= learningRate * biasGradient;
            for (let k = 0; k < net[1][i][j].length; k++) { // For each weight in that node
                const weightGradient = biasGradient * prevActivation[k];
                net[1][i][j][k] -= learningRate * weightGradient; 
            }
        }
        gradient = newGradientArr;
    }
}