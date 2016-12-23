package com.josephcatrambone.cg4j

import sun.plugin.dom.exception.InvalidStateException
import java.util.*

/***
 * Graph is little more than a node wrangler.
 * It helps set up the compute and push inputs/outputs, but you can call node forward/adjoint directly if you need to.
 * It also will handle saving and serializing.
 */
class Graph {
	var lastId: Int = 0
	var nodes = ArrayList<Node>() // Could be 'mutableList' in Kotlin, but I want to be sure it's not actually a list.

	fun getOutput(output: Node, inputSet: Map<Node, Tensor>): Tensor {
		return forward(output, inputSet)[output]!!
	}

	fun forward(output: Node, inputSet: Map<Node, Tensor>, cache: MutableMap<Node, Tensor> = mutableMapOf<Node, Tensor>() ): Map<Node, Tensor> {
		// Handle the input-node case implicitly.
		if(output in inputSet) {
			cache[output] = inputSet[output]!!
		}

		// Are we done?
		if(output !in cache) {
			// Nope.  Recursively solve for the inputs.
			var inputValues: Array<Tensor> = arrayOf()
			for (i in output.inputs) {
				forward(i, inputSet, cache) // Should calculate the cache.
				inputValues = inputValues.plus(cache[i]!!)
			}
			val result = output.forwardOperation(*inputValues)
			cache[output] = result
		}
		return cache
	}

	fun reverse(output: Node, inputSet: Map<Node, Tensor>, activations: Map<Node, Tensor>, adjointCache: MutableMap<Node, Tensor> = mutableMapOf<Node, Tensor>()): Map<Node, Tensor> {
		// If the cache is empty, we want to assign this to all ones.
		if(adjointCache.isEmpty()) {
			adjointCache[output] = Tensor.ones(*output.shape) // This should probably be a 1x1.
		}

		// Calculate the adjoint values of the inputs to this function, then recurse into them.
		// First, calculate the adjoint of each child.
		// Make an arrayList of Tensors from the inputs to this node.
		val forwardArguments : MutableList<Tensor> = mutableListOf() // Note: can't do `activations.filterKeys { nodeId -> nodeId in inputs[output] }.values.toTypedArray()` because we may get dupes in args.
		for(nodeId in output.inputs) {
			forwardArguments.add(activations[nodeId]!!)
		}
		val fwd = forwardArguments.toTypedArray()

		// Calculate updates.
		var adjointUpdates : Array<Tensor> = output.adjointOperation(fwd, adjointCache[output]!!)

		// Apply updates.
		for((nodeIndex, adjointValue) in output.inputs.zip(adjointUpdates)) {
			if(nodeIndex !in adjointCache) {
				adjointCache[nodeIndex] = adjointValue.dup()
			} else {
				adjointCache[nodeIndex]!!.add_i(adjointValue)
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
}