package com.josephcatrambone.cg4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import sun.plugin.dom.exception.InvalidStateException
import org.nd4j.linalg.ops.transforms.Transforms.*

//import org.nd4j.linalg.api.ops.impl.transforms.Tanh

class Graph {
	var shapes = arrayOf<IntArray>()
	var inputs = arrayOf<IntArray>()
	var operations = arrayOf< (Array<INDArray>)->INDArray >() // Node operations
	var adjointOperations = arrayOf< (Array<INDArray>, INDArray)->Array<INDArray> >() // Operation from forward activation + parent adjoint -> child adjoint modifications.

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

	// And now add the actual node types.

	fun addInput(vararg shape : Int): Int {
		shapes = shapes.plus(shape)
		inputs = inputs.plus(intArrayOf())
		operations = operations.plus( { x -> throw InvalidStateException("The 'operation' method was called on an input, which means it was unspecified in the input stream.") } )
		adjointOperations = adjointOperations.plus( { a, b -> arrayOf() } )
		return shapes.size-1
	}

	// NOTE: THESE MAY BE MADE OBSOLETE BY THE BROADCAST OPERATOR
	fun addAddConstant(left: Int, cons: Float): Int {
		shapes = shapes.plus(shapes[left])
		inputs = inputs.plus(intArrayOf(left))
		operations = operations.plus({ x -> x[0].add(cons) })
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint -> arrayOf(parentAdjoint) })
		return shapes.size-1
	}

	fun addMultiplyConstant(left: Int, cons: Float): Int {
		shapes = shapes.plus(shapes[left])
		inputs = inputs.plus(intArrayOf(left))
		operations = operations.plus({ x -> x[0].mul(cons) })
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint -> arrayOf(parentAdjoint.mul(cons)) })
		return shapes.size-1
	}
	// END

	fun addAdd(left: Int, right: Int): Int {
		shapes = shapes.plus(shapes[left])
		inputs = inputs.plus(intArrayOf(left, right))
		operations = operations.plus({ x -> x[0].add(x[1]) })
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint -> arrayOf(parentAdjoint, parentAdjoint) })
		return shapes.size-1
	}

	fun addSubtract(left: Int, right: Int): Int {
		shapes = shapes.plus(shapes[left])
		inputs = inputs.plus(intArrayOf(left, right))
		operations = operations.plus({ x ->
			x[0].sub(x[1])
		})
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint ->
			arrayOf(
				parentAdjoint,
				parentAdjoint.neg()
			)
		})
		return shapes.size-1
	}

	fun addElementMultiply(left: Int, right: Int): Int {
		shapes = shapes.plus(shapes[left])
		inputs = inputs.plus(intArrayOf(left, right))
		operations = operations.plus({ x ->
			val a = x[0]
			val b = x[1]
			a.mul(b)
		})
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint ->
			arrayOf(
				forwards[1].mul(parentAdjoint),
				forwards[0].mul(parentAdjoint)
			)
		})
		return shapes.size-1
	}

	fun addMatrixMultiply(left: Int, right: Int): Int {
		shapes = shapes.plus(intArrayOf(shapes[left][0], shapes[right][1]))
		inputs = inputs.plus(intArrayOf(left, right))
		operations = operations.plus({ x -> x[0].mmul(x[1]) })
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint ->
			arrayOf(
				// Left adjoint. If C=AB, adj(a) = adj(c)*bT
				parentAdjoint.mmul(forwards[1].transpose()),
				// Right adjoint.  adj(b) = aT*adj(c)
				forwards[0].transpose().mmul(parentAdjoint)
			)
		})
		return shapes.size-1
	}

	fun addTanh(operand: Int): Int {
		shapes = shapes.plus(shapes[operand].copyOf())
		inputs = inputs.plus(intArrayOf(operand))
		operations = operations.plus({ x -> tanh(x[0]) })
		// y := EW(x, op) -> y = tanh(x)
		// x_adj += EW(y_adj, EW(x, d_op), dot)
		// f(x) = tanh(x).  df(x) = 1 - tanh(x)^2
		// x_adj += y_adj * d_op(x)
		adjointOperations = adjointOperations.plus({ forwards, parentAdjoint ->
			arrayOf(
				// adjoint[operand][i] += adjoint[node][i]*(1.0f - (tanh(forward[operand][i])*tanh(forward[operand][i])));
				//parentAdjoint.mul(Nd4j.onesLike(parentAdjoint).sub(forwards[thisId].mul(forwards[thisId])))
				parentAdjoint.mul(Nd4j.onesLike(parentAdjoint).sub(tanh(forwards[0]).mul(tanh(forwards[0]))))
			)
		})
		return shapes.size-1
	}

	fun addAddWithBroadcast(targetNodeToMatch: Int, operandToBroadcast: Int): Int {
		shapes = shapes.plus(shapes[targetNodeToMatch].copyOf())
		inputs = inputs.plus(intArrayOf(targetNodeToMatch, operandToBroadcast))
		operations = operations.plus({ x -> x[1]!!.repmat(*x[0]!!.shape()) })
		adjointOperations = adjointOperations.plus({ forwards, thisAdjoint ->
			throw NotImplementedError("Still working on backprop with broadcast.")
			arrayOf(
				// TODO: Start here.
			)
		})
		return shapes.size-1
	}

	fun addPower(base: Int, exp: Float): Int { // x^c is supported.  c^x is not yet supported.
		shapes = shapes.plus(shapes[base].copyOf())
		inputs = inputs.plus(intArrayOf(base))
		operations = operations.plus({ x -> pow(x[0]!!, exp)})
		// x_adj += EW(y_adj, EW(x, d_op), dot)
		// d/dx 1/x = d/dx x^-1 = -(x^-2) = -(1/x^2)
		adjointOperations = adjointOperations.plus({ forwards, thisAdjoint ->
			arrayOf(
					thisAdjoint.mul(pow(forwards[0], exp-1.0f).mul(exp))
			)
		})
		return shapes.size - 1
	}

	fun addAbs(oper: Int): Int {
		shapes = shapes.plus(shapes[oper].copyOf())
		inputs = inputs.plus(intArrayOf(oper))
		operations = operations.plus({ x -> abs(x[0]!!) })
		// x_adj += EW(y_adj, EW(x, d_op), dot)
		adjointOperations = adjointOperations.plus({ forwards, thisAdjoint ->
			arrayOf(
				thisAdjoint.mul(sign(forwards[0]))
			)
		})
		return shapes.size - 1
	}

	fun addConvolution(inputMatrix: Int, convolutionKernel: Int): Int {
		// TODO: Implement.
		throw NotImplementedError("Need to work on convolution.")
		return -1
	}
}