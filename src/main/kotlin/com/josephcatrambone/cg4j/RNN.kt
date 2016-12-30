package com.josephcatrambone.cg4j

import com.sun.prism.paint.Gradient
import java.util.*

/**
 * Created by jcatrambone on 12/19/16.
 */
class RNN(val inputSize: Int, val hiddenSize: Int, startWeightScale:Float = 0.1f) {
	var weightNode_ih: VariableNode = VariableNode(inputSize, hiddenSize)
	var weightNode_hh: VariableNode = VariableNode(hiddenSize, hiddenSize)
	var weightNode_ho: VariableNode = VariableNode(hiddenSize, inputSize)
	var biasNode_h: VariableNode = VariableNode(hiddenSize)
	var biasNode_o: VariableNode = VariableNode(inputSize)

	init {
		val random = Random()
		weightNode_ih.value.elementOperation_i { i -> random.nextGaussian().toFloat()*startWeightScale }
		weightNode_hh.value.elementOperation_i { i -> random.nextGaussian().toFloat()*startWeightScale }
		weightNode_ho.value.elementOperation_i { i -> random.nextGaussian().toFloat()*startWeightScale }
	}

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
		val hiddenStartNode = InputNode(1, hiddenSize)
		var previousHiddenValue : Node = hiddenStartNode
		for(step in (0..x.shape[0]-1)) {
			val inNode = InputNode(1, inputSize)
			val nextHiddenState = SigmoidNode(
				AddNode(
					AddNode(
						MatrixMultiplyNode(inNode, weightNode_ih),
						MatrixMultiplyNode(previousHiddenValue, weightNode_hh)
					),
					biasNode_h
				)
			)
			val outNode = TanhNode(AddNode(MatrixMultiplyNode(nextHiddenState, weightNode_ho), biasNode_o))
			inputNodes.add(inNode)
			previousHiddenValue = GradientClipNode(nextHiddenState)
			outputNodes.add(outNode)

			// Calculate the error for this timestep.
			val target = InputNode(inputSize)
			val error = AbsNode(SubtractNode(outNode, target))
			targetNodes.add(target)
			errorNodes.add(error)

			graph.add(error)
			// TODO: Inject a gradient normalization node in here.
		}

		// Zip everything together and calculate the gradient.
		val inputMap = mutableMapOf<Node, Tensor>()
		inputMap[hiddenStartNode] = Tensor(shape=intArrayOf(1, hiddenSize), data=hiddenState)
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
		biasNode_h.value.sub_i(grad[biasNode_h]!!.mul(learningRate))
		biasNode_o.value.sub_i(grad[biasNode_o]!!.mul(learningRate))
	}

	// TODO: This returns a list of floats until we get Tensor vertical concat working.
	fun predict(maxSteps:Int, inputTensor:Tensor? = null, initialHiddenState: FloatArray = FloatArray(size = hiddenSize)): Pair<FloatArray, Array<FloatArray>> {
		val outputArray = mutableListOf<FloatArray>()

		// Build our graph.
		val graph = Graph()
		val inputNode = InputNode(inputSize)
		val prevHiddenNode = InputNode(hiddenSize)

		val nextHiddenState = SigmoidNode(
			AddNode(
				AddNode(
					MatrixMultiplyNode(inputNode, weightNode_ih),
					MatrixMultiplyNode(prevHiddenNode, weightNode_hh)
				),
				biasNode_h
			)
		)
		val outNode = TanhNode(AddNode(MatrixMultiplyNode(nextHiddenState, weightNode_ho), biasNode_o))
		graph.add(outNode)

		// Set up initial tensors.
		val inputMap = mutableMapOf<Node, Tensor>()
		var hiddenState = Tensor(shape=intArrayOf(1, hiddenSize), data=initialHiddenState)
		var previousOutput = Tensor.zeros(1, inputSize)
		for(i in (0..maxSteps)) {
			inputMap[prevHiddenNode] = hiddenState
			if(inputTensor == null || i >= inputTensor.shape[0]) {
				inputMap[inputNode] = previousOutput
			} else {
				inputMap[inputNode] = inputTensor.getSubtensor(0, i)
			}

			val fwd = graph.forward(outNode, inputMap)
			hiddenState = fwd[nextHiddenState]!!
			previousOutput = fwd[outNode]!!
			outputArray.add(previousOutput.data)
		}

		return Pair(hiddenState.data, outputArray.toTypedArray())
	}
}