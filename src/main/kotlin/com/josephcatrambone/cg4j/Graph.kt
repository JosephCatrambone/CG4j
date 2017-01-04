package com.josephcatrambone.cg4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.File
import java.util.*

/***
 * Graph is little more than a node wrangler.
 * It helps set up the compute and push inputs/outputs, but you can call node forward/adjoint directly if you need to.
 * It also will handle saving and serializing.
 */
class Graph {
	var lastId: Int = 0
	var nodes = ArrayList<Node>() // Could be 'mutableList' in Kotlin, but I want to be sure it's not actually a list.

	fun getOutput(output: Node, inputSet: Map<Node, INDArray>): INDArray {
		return forward(output, inputSet)[output]!!
	}

	fun forward(output: Node, inputSet: Map<Node, INDArray>, cache: MutableMap<Node, INDArray> = mutableMapOf<Node, INDArray>() ): Map<Node, INDArray> {
		// Handle the input-node case implicitly.
		if(output in inputSet) {
			cache[output] = inputSet[output]!!
		}

		// Are we done?
		if(output !in cache) {
			// Nope.  Recursively solve for the inputs.
			var inputValues: Array<INDArray> = arrayOf()
			for (i in output.inputs) {
				forward(i, inputSet, cache) // Should calculate the cache.
				inputValues = inputValues.plus(cache[i]!!)
			}
			val result = output.forwardOperation(*inputValues)
			cache[output] = result
		}
		return cache
	}

	fun reverse(output: Node, inputSet: Map<Node, INDArray>, activations: Map<Node, INDArray>, adjointCache: MutableMap<Node, INDArray> = mutableMapOf<Node, INDArray>()): Map<Node, INDArray> {
		// If the cache is empty, we want to assign this to all ones.
		if(adjointCache.isEmpty()) {
			adjointCache[output] = Nd4j.ones(*output.shape) // This should probably be a 1x1.
		}

		// Calculate the adjoint values of the inputs to this function, then recurse into them.
		// First, calculate the adjoint of each child.
		// Make an arrayList of Tensors from the inputs to this node.
		val forwardArguments : MutableList<INDArray> = mutableListOf() // Note: can't do `activations.filterKeys { nodeId -> nodeId in inputs[output] }.values.toTypedArray()` because we may get dupes in args.
		for(nodeId in output.inputs) {
			forwardArguments.add(activations[nodeId]!!)
		}
		val fwd = forwardArguments.toTypedArray()

		// Calculate updates.
		var adjointUpdates : Array<INDArray> = output.adjointOperation(fwd, adjointCache[output]!!)

		// Apply updates.
		for((nodeIndex, adjointValue) in output.inputs.zip(adjointUpdates)) {
			if(nodeIndex !in adjointCache) {
				adjointCache[nodeIndex] = adjointValue.dup()
			} else {
				adjointCache[nodeIndex]!!.addi(adjointValue)
			}
		}

		// Now update all the children with this adjoint value.
		for(childId in output.inputs) {
			reverse(childId, inputSet, activations, adjointCache)
		}

		return adjointCache
	}

	fun add(node: Node): Node {
		// First, add the inputs to the graph recursively _if_ they're not already present.
		// This can be an expensive operation, but we only build the graph once, so we're okay.
		for(n in node.inputs) {
			if(!nodes.contains(n)) {
				add(n)
			}
		}
		node.id = lastId
		lastId++
		nodes.add(node)
		return node
	}

	fun save(f: File) {
		val fout = f.bufferedWriter()
		try {
			nodes.forEach { n ->
				//fout.write(n.javaClass.canonicalName) //-> com.josephcatrambone.cg4j.something
				fout.write(n.javaClass.simpleName) //->something
				fout.write("|")
				fout.write(n.id.toString())
				fout.write("|")
				fout.write(n.name)
				fout.write("|")
				fout.write(n.inputs.map { input -> input.id }.joinToString(separator=","))
				fout.write("|")
				fout.write(n.shape.joinToString(separator=","))
				fout.write("|")
				fout.write(n.extraDataToString(separator="|"))
				fout.newLine()
			}
		} finally{
			fout.close()
		}
	}

	fun load(f: File) {
		val fout = f.bufferedReader()
		try {
			nodes = ArrayList()
			fout.forEachLine {
				val tokens = it.split("|").iterator()
				//n.javaClass.canonicalName)
				var className = tokens.next()
				// If we used canonical names, we could do this: var n : Node = Class.forName(className).getConstructor().newInstance() as Node
				//n.id.toString())
				val id =  tokens.next().toInt()
				//fout.write(n.name)
				val name = tokens.next()
				//fout.write(n.inputs.map { input -> input.id }.joinToString(separator=","))
				val inputToken = tokens.next()
				var inputs = arrayOf<Node>()
				if(inputToken != "") {
					val inputList = inputToken.split(",")
					inputs = inputList.map { inId -> this.nodes[inId.toInt()] }.toTypedArray()
				}
				//fout.write(n.shape.joinToString(separator=","))
				val shapeList = tokens.next().split(",")
				val shape = shapeList.map{dimensionValue -> dimensionValue.toInt()}.toIntArray()
				//fout.write(n.extraDataToString())
				val n = when(className) {
					"InputNode" -> InputNode(*shape)
					"VariableNode" -> InputNode(*shape)
					"AddConstantNode" -> AddConstantNode(inputs[0], -1f)
					"AddNode" -> AddNode(inputs[0], inputs[1])
					"SubtractNode" -> SubtractNode(inputs[0], inputs[1])
					"ConstantMultiplyNode" -> ConstantMultiplyNode(inputs[0], -1f)
					"ElementMultiplyNode" -> ElementMultiplyNode(inputs[0], inputs[1])
					"MatrixMultiplyNode" -> MatrixMultiplyNode(inputs[0], inputs[1])
					"TanhNode" -> TanhNode(inputs[0])
					"SigmoidNode" -> SigmoidNode(inputs[0])
					"LeakyReLUNode" -> LeakyReLUNode(inputs[0])
					"PowerNode" -> PowerNode(inputs[0], -1f)
					"AbsNode" -> AbsNode(inputs[0])
					else -> throw RuntimeException("Unrecognized class type to deserialize: $className")
				}
				n.id = id
				n.name = name
				n.inputs = inputs
				n.shape = shape
				n.extraDataFromStringIterator(tokens)
				nodes.add(n)
				assert(nodes.size == n.id)
			}
		} finally{
			fout.close()
		}
	}
}