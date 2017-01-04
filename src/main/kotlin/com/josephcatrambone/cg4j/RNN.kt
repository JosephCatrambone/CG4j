package com.josephcatrambone.cg4j

import com.sun.prism.paint.Gradient
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.util.*

/**
 * Created by jcatrambone on 12/19/16.
 */
class RNN(val inputSize: Int, val hiddenSize: Int, startWeightScale:Float = 0.1f) {
	var weightNode_ih: VariableNode = VariableNode(inputSize, hiddenSize)
	var weightNode_hh: VariableNode = VariableNode(hiddenSize, hiddenSize)
	var weightNode_ho: VariableNode = VariableNode(hiddenSize, inputSize)
	var biasNode_h: VariableNode = VariableNode(1, hiddenSize)
	var biasNode_o: VariableNode = VariableNode(1, inputSize)

	/*
	i = sig(x_i*U_i + s_t-1*W_i)
	f = sig(x_t*U_f + s_t-1*W_f)
	o = sig(x_t*U_o + s_t-1*W_o)
	g = tanh(x_t*U_g + s_t-1*W_g)
	c_t = c_t-1 dot f + g dot i
	s_t = tanh(c_t) dot o
	 */

	init {
		val random = Random()
		/*
		weightNode_ih.value.elementOperation_i { i -> random.nextGaussian().toFloat()*startWeightScale }
		weightNode_hh.value.elementOperation_i { i -> random.nextGaussian().toFloat()*startWeightScale }
		weightNode_ho.value.elementOperation_i { i -> random.nextGaussian().toFloat()*startWeightScale }
		*/
	}

	/*** fit
	 * Build a graph and unwrap the connections to fit x to y.
	 * @param x A tensor of shape [sequence length, input size]
	 * @param y A tensor of shape [sqeuence length, input size]
	 * @param hiddenState The previous hidden state to use.  Defaults to a zero vector.
	 */
	fun fit(x:INDArray, y:INDArray, learningRate:Float, hiddenState:FloatArray = FloatArray(size=hiddenSize)) {
		val graph = Graph()
		val inputNodes = mutableListOf<Node>()
		val outputNodes = mutableListOf<Node>()
		val targetNodes = mutableListOf<Node>()
		val errorNodes = mutableListOf<Node>()

		// Unroll everything
		val hiddenStartNode = InputNode(1, hiddenSize)
		var previousHiddenValue : Node = hiddenStartNode
		for(step in (0..x.rows())) {
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
			previousHiddenValue = nextHiddenState
			outputNodes.add(outNode)

			// Calculate the error for this timestep.
			val target = InputNode(1, inputSize)
			val error = AbsNode(SubtractNode(outNode, target))
			targetNodes.add(target)
			errorNodes.add(error)
			// TODO: Inject a gradient normalization node in here.
		}

		// Zip everything together and calculate the gradient.
		val inputMap = mutableMapOf<Node, INDArray>()
		inputMap[hiddenStartNode] = Nd4j.create(hiddenState, intArrayOf(1, hiddenSize))
		for(i in (0..inputNodes.size-1)) {
			inputMap[inputNodes[i]] = x.getRow(i)
			inputMap[targetNodes[i]] = y.getRow(i)
		}

		// Make one final output operation which we will minimize.
		val totalError = AddNode(*errorNodes.toTypedArray())
		graph.add(totalError)

		// Grads
		val fwd = graph.forward(totalError, inputMap)
		val grad = graph.reverse(totalError, inputMap, fwd)

		// Apply gradients!
		weightNode_ih.value.subi(grad[weightNode_ih]!!.mul(learningRate))
		weightNode_hh.value.subi(grad[weightNode_hh]!!.mul(learningRate))
		weightNode_ho.value.subi(grad[weightNode_ho]!!.mul(learningRate))
		biasNode_h.value.subi(grad[biasNode_h]!!.mul(learningRate))
		biasNode_o.value.subi(grad[biasNode_o]!!.mul(learningRate))
	}

	// TODO: This returns a list of floats until we get Tensor vertical concat working.
	fun predict(maxSteps:Int, inputTensor:INDArray? = null, initialHiddenState: FloatArray = FloatArray(size = hiddenSize)): Pair<FloatArray, Array<FloatArray>> {
		val outputArray = mutableListOf<FloatArray>()

		// Build our graph.
		val graph = Graph()
		val inputNode = InputNode(1, inputSize)
		val prevHiddenNode = InputNode(1, hiddenSize)

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
		val inputMap = mutableMapOf<Node, INDArray>()
		var hiddenState = Nd4j.create(initialHiddenState, intArrayOf(1, hiddenSize))
		var previousOutput = Nd4j.zeros(1, inputSize)
		for(i in (0..maxSteps)) {
			inputMap[prevHiddenNode] = hiddenState
			if(inputTensor == null || i >= inputTensor.rows()) {
				inputMap[inputNode] = previousOutput
			} else {
				inputMap[inputNode] = inputTensor.getRow(i)
			}

			val fwd = graph.forward(outNode, inputMap)
			hiddenState = fwd[nextHiddenState]!!
			previousOutput = fwd[outNode]!!
			outputArray.add(previousOutput.data().asFloat())
		}

		return Pair(hiddenState.data().asFloat(), outputArray.toTypedArray())
	}
}