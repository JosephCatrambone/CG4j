package com.josephcatrambone.cg4j

/**
 * Created by jcatrambone on 12/19/16.
 */
class RNN(val inputSize: Int, val hiddenSize: Int) {
	var weightNode_ih: VariableNode = VariableNode(inputSize, hiddenSize)
	var weightNode_hh: VariableNode = VariableNode(hiddenSize, hiddenSize)
	var weightNode_ho: VariableNode = VariableNode(hiddenSize, inputSize)
	var biasNode: VariableNode = VariableNode(hiddenSize)

	/*** fit
	 * Build a graph and unwrap the connections to fit x to y.
	 * @param x A tensor of shape [sequence length, input size]
	 * @param y A tensor of shape [sqeuence length, input size]
	 * @param hiddenState The previous hidden state to use.  Defaults to a zero vector.
	 */
	fun fit(x:Tensor, y:Tensor, learningRate:Float, hiddenState:FloatArray = FloatArray(size=hiddenSize)) {
		val graph = Graph()
		val inputNodes = mutableListOf<Node>()
		val outputNodes = mutableListOf<Node>()
		val targetNodes = mutableListOf<Node>()
		val errorNodes = mutableListOf<Node>()

		// Unroll everything
		val hiddenStartNode = InputNode(hiddenSize)
		var previousHiddenValue : Node = hiddenStartNode
		for(step in (0..x.shape[0]-1)) {
			val inNode = InputNode(inputSize)
			val nextHiddenState = TanhNode(AddNode(MatrixMultiplyNode(inNode, weightNode_ih), MatrixMultiplyNode(previousHiddenValue, weightNode_hh)))
			val outNode = TanhNode(MatrixMultiplyNode(nextHiddenState, weightNode_ho))
			inputNodes.add(inNode)
			previousHiddenValue = nextHiddenState
			outputNodes.add(outNode)

			// Calculate the error for this timestep.
			val target = InputNode(inputSize)
			val error = AbsNode(SubtractNode(outNode, target))
			targetNodes.add(target)
			errorNodes.add(error)

			// TODO: Inject a gradient normalization node in here.
		}

		// Zip everything together and calculate the gradient.
		val inputMap = mutableMapOf<Node, Tensor>()
		inputMap[hiddenStartNode] = Tensor(shape=intArrayOf(hiddenSize), data=hiddenState)
		for(i in (0..inputNodes.size-1)) {
			inputMap[inputNodes[i]] = x.getSubtensor(0, i)
			inputMap[targetNodes[i]] = y.getSubtensor(0, i)
		}

		// Grads
		val fwd = graph.forward(errorNodes.last(), inputMap)
		val grad = graph.reverse(errorNodes.last(), inputMap, fwd)

		// Apply gradients!
		weightNode_ih.value.sub_i(grad[weightNode_ih]!!.mul(learningRate))
		weightNode_hh.value.sub_i(grad[weightNode_hh]!!.mul(learningRate))
		weightNode_ho.value.sub_i(grad[weightNode_ho]!!.mul(learningRate))
	}
}