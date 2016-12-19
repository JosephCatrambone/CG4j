package com.josephcatrambone.cg4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.convolution.Convolution
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import sun.plugin.dom.exception.InvalidStateException

//import org.nd4j.linalg.api.ops.impl.transforms.Tanh

class Graph {
	var buildInstructions = mutableListOf<String>()
	var shapes = arrayOf<IntArray>()
	var inputs = arrayOf<IntArray>()
	var variables = mutableMapOf<Int, INDArray>()
	var operations = arrayOf< (Array<INDArray>)->INDArray >() // Node operations
	var adjointOperations = arrayOf< (Array<INDArray>, INDArray)->Array<INDArray> >() // Operation from forward activation + parent adjoint -> child adjoint modifications.

	fun toText(): String {
		val sb = StringBuilder()
		// Add build instructions.
		buildInstructions.forEach { s -> sb.append(s); sb.append("\n") }
		// Save the variables.
		sb.append("variables")
		variables.forEach { ent -> sb.append(ent.key); sb.append("\n"); sb.append(ent.value.toString()) }
		return ""
	}

	fun fromText(net: String) {

	}

	fun getOutput(output: Int, inputSet: Map<Int, INDArray>): INDArray {
		return forward(output, inputSet)[output]!!
	}

	fun forward(output: Int, inputSet: Map<Int, INDArray>, cache: MutableMap<Int, INDArray> = mutableMapOf<Int, INDArray>() ): Map<Int, INDArray> {
		// Handle the input-node case implicitly.
		if(output in inputSet) {
			cache[output] = inputSet[output]!!
		}

		// Are we done?
		if(output !in cache) {
			// Nope.  Recursively solve for the inputs.
			var inputValues: Array<INDArray> = arrayOf()
			for (i in inputs[output]) {
				forward(i, inputSet, cache) // Should calculate the cache.
				inputValues = inputValues.plus(cache[i]!!)
			}
			val result = operations[output](inputValues)
			cache[output] = result
		}
		return cache
	}

	fun reverse(output: Int, inputSet: Map<Int, INDArray>, activations: Map<Int, INDArray>, adjointCache: MutableMap<Int, INDArray> = mutableMapOf<Int, INDArray>()): Map<Int, INDArray> {
		// If the cache is empty, we want to assign this to all ones.
		if(adjointCache.isEmpty()) {
			adjointCache[output] = Nd4j.ones(*shapes[output]) // This should probably be a 1x1.
		}

		// Calculate the adjoint values of the inputs to this function, then recurse into them.
		// First, calculate the adjoint of each child.
		// Make an arrayList of INDArrays from the inputs to this node.
		val forwardArguments : MutableList<INDArray> = mutableListOf() // Note: can't do `activations.filterKeys { nodeId -> nodeId in inputs[output] }.values.toTypedArray()` because we may get dupes in args.
		for(nodeId in inputs[output]) {
			forwardArguments.add(activations[nodeId]!!)
		}
		val fwd = forwardArguments.toTypedArray()

		// Calculate updates.
		var adjointUpdates : Array<INDArray> = adjointOperations[output](fwd, adjointCache[output]!!)

		// Apply updates.
		for((nodeIndex, adjointValue) in inputs[output].zip(adjointUpdates)) {
			if(nodeIndex !in adjointCache) {
				adjointCache[nodeIndex] = adjointValue.dup()
			} else {
				adjointCache[nodeIndex]!!.addi(adjointValue)
			}
		}

		// Now update all the children with this adjoint value.
		for(childId in inputs[output]) {
			reverse(childId, inputSet, activations, adjointCache)
		}

		return adjointCache
	}

	fun addNode(instruction: String, shape: IntArray, input: IntArray, operation: (Array<INDArray>)->INDArray, adjoint: (Array<INDArray>, INDArray)->Array<INDArray>): Int {
		this.buildInstructions.add(instruction)
		this.shapes = this.shapes.plus(shape)
		this.inputs = this.inputs.plus(input)
		this.operations = this.operations.plus(operation)
		this.adjointOperations = this.adjointOperations.plus(adjoint)
		return shapes.size-1
	}

	// And now add the actual node types.

	fun input(vararg shape : Int): Int {
		return addNode(
			"input",
			shape,
			intArrayOf(),
			{ x -> throw InvalidStateException("The 'operation' method was called on an input, which means it was unspecified in the input stream.") },
			{ a, b -> arrayOf() }
		)
	}

	fun variable(vararg shape: Int): Int {
		val variableIndex = shapes.size // Force the shapes to match up with the variable index.
		variables[variableIndex] = Nd4j.zeros(*shape)
		return addNode(
			"variable",
			shape,
			intArrayOf(),
			{ x -> this.variables[variableIndex]!! },
			{ a, b -> arrayOf() }
		)
	}

	// NOTE: THESE MAY BE MADE OBSOLETE BY THE BROADCAST OPERATOR
	fun constantAdd(left: Int, cons: Float): Int {
		return addNode(
			"constant",
			shapes[left],
			intArrayOf(left),
			{ x -> x[0].add(cons) },
			{ forwards, parentAdjoint -> arrayOf(parentAdjoint) }
		)
	}

	fun constantMultiply(left: Int, cons: Float): Int {
		return addNode(
			"multiplyConstant",
			shapes[left],
			intArrayOf(left),
			{ x -> x[0].mul(cons) },
			{ forwards, parentAdjoint -> arrayOf(parentAdjoint.mul(cons)) }
		)
	}
	// END

	fun add(left: Int, right: Int): Int {
		return addNode(
			"add",
			shapes[left],
			intArrayOf(left, right),
			{ x -> x[0].add(x[1]) },
			{ forwards, parentAdjoint -> arrayOf(parentAdjoint, parentAdjoint) }
		)
	}

	fun subtract(left: Int, right: Int): Int {
		return addNode(
			"subtract",
			shapes[left],
			intArrayOf(left, right),
			{ x -> x[0].sub(x[1]) },
			{ forwards, parentAdjoint ->
				arrayOf(
					parentAdjoint,
					parentAdjoint.neg()
				)
			}
		)
	}

	fun elementMultiply(left: Int, right: Int): Int {
		return addNode(
			"elementMultiply",
			shapes[left],
			intArrayOf(left, right),
			{ x -> x[0].mul(x[1]) },
			{ forwards, parentAdjoint ->
				arrayOf(
					forwards[1].mul(parentAdjoint),
					forwards[0].mul(parentAdjoint)
				)
			}
		)
	}

	fun matrixMultiply(left: Int, right: Int): Int {
		return addNode(
			"matrixMultiply",
			intArrayOf(shapes[left][0], shapes[right][1]),
			intArrayOf(left, right),
			{ x -> x[0].mmul(x[1]) },
			{ forwards, parentAdjoint ->
				arrayOf(
					// Left adjoint. If C=AB, adj(a) = adj(c)*bT
					parentAdjoint.mmul(forwards[1].transpose()),
					// Right adjoint.  adj(b) = aT*adj(c)
					forwards[0].transpose().mmul(parentAdjoint)
				)
			}
		)
	}

	fun tanh(operand: Int): Int {
		return addNode(
			"tanh",
			shapes[operand].copyOf(),
			intArrayOf(operand),
			{ x -> Transforms.tanh(x[0]) },
		// y := EW(x, op) -> y = tanh(x)
		// x_adj += EW(y_adj, EW(x, d_op), dot)
		// f(x) = tanh(x).  df(x) = 1 - tanh(x)^2
		// x_adj += y_adj * d_op(x)
			{ forwards, parentAdjoint ->
				arrayOf(
					// adjoint[operand][i] += adjoint[node][i]*(1.0f - (tanh(forward[operand][i])*tanh(forward[operand][i])));
					//parentAdjoint.mul(Nd4j.onesLike(parentAdjoint).sub(forwards[thisId].mul(forwards[thisId])))
					parentAdjoint.mul(Nd4j.onesLike(parentAdjoint).sub(Transforms.tanh(forwards[0]).mul(Transforms.tanh(forwards[0]))))
				)
			}
		)
	}

	fun addWithBroadcast(targetNodeToMatch: Int, operandToBroadcast: Int): Int {
		return addNode(
			"addWithBroadcast",
			shapes[targetNodeToMatch].copyOf(),
			intArrayOf(targetNodeToMatch, operandToBroadcast),
			{ x -> x[1].repmat(*x[0].shape()) },
			{ forwards, thisAdjoint ->
				throw NotImplementedError("Still working on backprop with broadcast.")
				arrayOf(
					// TODO: Start here.
				)
			}
		)
	}

	fun power(base: Int, exp: Float): Int { // x^c is supported.  c^x is not yet supported.
		return addNode(
			"power",
			shapes[base].copyOf(),
			intArrayOf(base),
			{ x -> Transforms.pow(x[0], exp) },
		// x_adj += EW(y_adj, EW(x, d_op), dot)
		// d/dx 1/x = d/dx x^-1 = -(x^-2) = -(1/x^2)
			{ forwards, thisAdjoint ->
				arrayOf(
					thisAdjoint.mul(Transforms.pow(forwards[0], exp-1.0f).mul(exp))
				)
			}
		)
	}

	fun abs(oper: Int): Int {
		return addNode(
			"abs",
			shapes[oper].copyOf(),
			intArrayOf(oper),
			{ x -> Transforms.abs(x[0]) },
		// x_adj += EW(y_adj, EW(x, d_op), dot)
			{ forwards, thisAdjoint ->
				arrayOf(
					thisAdjoint.mul(Transforms.sign(forwards[0]))
				)
			}
		)
	}

	fun convolution2D(inputMatrix: Int, convolutionKernel: Int, stride: Int): Int {
		// Input: Volume of W1 H1 D1
		// Params: K -> # filters, F -> Spatial extent, S -> Stride, P -> Padding.
		// Output: W2 = (W1 - F + 2P)/S + 1 || H2 is like W2 || D2 = K
		return addNode(
			"conv2D",
			intArrayOf(), // TODO:
			intArrayOf(inputMatrix, convolutionKernel),
			{ x -> Convolution.conv2d(x[0], x[1], Convolution.Type.SAME) },
			{ forwards, thisAdjoint -> arrayOf() } // TODO:
		)
	}

	fun deconvolution2D(inputMatrix: Int, deconvolutionKernel: Int, stride: Int): Int {
		return -1
	}
}