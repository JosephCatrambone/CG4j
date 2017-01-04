package com.josephcatrambone.cg4j

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms

/**
 * Created by jcatrambone on 12/22/16.
 * Some quick notes about approach:
 * If we have input as an array of IDs (where each ID refers to a graph node), then...
 *  + We can serialize very easily.
 *  - We have to pass a reference to the graph when we create the nodes, otherwise they can't figure out size.
 * If we have inputs as an array of Nodes, then...
 *  + We can easily determine the size of the nodes.
 *  + When we do the compute w/ memoization, we automatically prune unused nodes.
 *  - Hard to serialize and unserialize.
 * If we have blank constructors, it's easier for us to just instance a new node of whatever class and fill it in later.
 * This functionality is delegated to the graph, since it's already thin and doesn't have much responsibility.
 * It means also we can do a better job handling the deserialization from node IDs.
 */
abstract class Node(var shape:IntArray=intArrayOf(), var inputs:Array<Node> =arrayOf<Node>()) {
	var id: Int = -1 // When we serialize, we write these IDs instead of recursively serializing objects.
	var name:String = ""
	abstract fun forwardOperation(vararg inputValues: INDArray): INDArray
	abstract fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray>

	// These are used for serialization.
	// By default, the graph will populate the id, name, shape, and inputs.
	// If a node needs extra data, like a stored value, that has to be put in the extra data.
	open fun extraDataToString(separator:String="|"): String { return "" }
	open fun extraDataFromStringIterator(it:Iterator<String>) {}
}

class InputNode(vararg shape:Int) : Node(shape) {
	//constructor() : this(-1) {} // Allow empty constructor when reloading from disk.  We'll set these later.

	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		throw RuntimeException("You should never see this.  The graph should be checking the inputs before this gets called.")
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf()
	}
}

class VariableNode(vararg shape:Int) : Node(shape) {
	var value: INDArray = Nd4j.zeros(*shape)

	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return value
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf()
	}

	override fun extraDataToString(separator:String):String {
		return value.toString()
	}

	override fun extraDataFromStringIterator(it: Iterator<String>) {
		//super.extraDataFromStringIterator(it)
		this.value = INDArray.fromString(it.next())
	}

}

class AddConstantNode(lhs:Node, var c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return inputValues[0].add(c)
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(adjoint)
	}

	override fun extraDataToString(separator:String):String {
		return c.toString()
	}

	override fun extraDataFromStringIterator(it: Iterator<String>) {
		//super.extraDataFromStringIterator(it)
		this.c = it.next().toFloat()
	}
}

class AddNode(vararg inputs:Node) : Node(inputs[0].shape, arrayOf<Node>(*inputs)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		val out:INDArray = Nd4j.zeros(*this.shape)
		for(t in inputValues) {
			out.addi(t)
		}
		return out
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		val adjointList = mutableListOf<INDArray>()
		for(fv in forwardValues) {
			adjointList.add(adjoint) // A copy of the adjoint ref.  Do we want to clone this?
		}
		return adjointList.toTypedArray()
	}
}

class SubtractNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return inputValues[0].sub(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(adjoint, adjoint.neg())
	}
}

class ConstantMultiplyNode(lhs:Node, var c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return inputValues[0].mul(c)
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(adjoint.mul(c))
	}

	override fun extraDataToString(separator:String):String {
		return c.toString()
	}

	override fun extraDataFromStringIterator(it: Iterator<String>) {
		this.c = it.next().toFloat()
	}
}

class ElementMultiplyNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return inputValues[0].mul(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(forwardValues[1].mul(adjoint), forwardValues[0].mul(adjoint))
	}
}

class MatrixMultiplyNode : Node {
	// TODO: Figure out how to better use the tensor product here.
	constructor(lhs:Node, rhs:Node) : super(shape = intArrayOf(lhs.shape[0], rhs.shape[1]), inputs = arrayOf<Node>(lhs, rhs)) {

	}

	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return inputValues[0].mmul(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(
			// Left adjoint. If C=AB, adj(a) = adj(c)*bT
			adjoint.mmul(forwardValues[1].transpose()),
			// Right adjoint.  adj(b) = aT*adj(c)
			forwardValues[0].transpose().mmul(adjoint)
		)
	}
}

class TanhNode(n:Node) : Node(shape=n.shape, inputs=arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return Transforms.tanh(inputValues[0])
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(
			adjoint.mul(Nd4j.ones(*adjoint.shape()).sub(Transforms.tanh(forwardValues[0]).mul(Transforms.tanh(forwardValues[0]))))
		)
	}
}

class SigmoidNode(n:Node) : Node(shape=n.shape, inputs=arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return Transforms.sigmoid(inputValues[0])
	}

	// Derivative of sigmoid = s * (1-s)
	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(
			adjoint.mul(Transforms.sigmoid(forwardValues[0]).mul(Nd4j.onesLike(forwardValues[0]).subi(Transforms.sigmoid(forwardValues[0]))))
		)
	}
}

class LeakyReLUNode(n:Node) : Node(n.shape, arrayOf<Node>(n)) {

	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		// Do slices along the given axis.
		return Transforms.leakyRelu(inputValues[0])
	}

	override fun adjointOperation(forwardValues: Array<INDArray>, adjoint: INDArray): Array<INDArray> {
		return arrayOf(
			forwardValues[0].elementOperation(adjoint, { x, adj -> if(x >= 0f) { adj } else {leak*adj} })
		)
	}
}

class PowerNode(base:Node, var exponent: Float) : Node(shape=base.shape, inputs=arrayOf<Node>(base)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return Transforms.pow(inputValues[0], exponent)
	}

	override fun adjointOperation(forwardValues:Array<INDArray>, adjoint:INDArray): Array<INDArray> {
		return arrayOf(
			adjoint.mul(Transforms.pow(forwardValues[0], (exponent-1.0f)).mul(exponent))
		)
	}

	override fun extraDataToString(separator:String):String {
		return exponent.toString()
	}

	override fun extraDataFromStringIterator(it: Iterator<String>) {
		//super.extraDataFromStringIterator(it)
		this.exponent = it.next().toFloat()
	}
}

class AbsNode(n:Node) : Node(n.shape, arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return Transforms.abs(inputValues[0])
	}

	override fun adjointOperation(forwardValues: Array<INDArray>, adjoint: INDArray): Array<INDArray> {
		return arrayOf(
			adjoint.mul(Transforms.sign(forwardValues[0]))
		)
	}
}

/*** SoftmaxNode
 * Performs softmax on the values of the last axis of the tensor.
 */
class SoftmaxNode(n:Node) : Node(n.shape, arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		val out = Nd4j.zerosLike(inputValues[0])
		var accumulator:Float = 0f
		var started = true
		for(outerIndex in 0..inputValues[0].data.size-1) { // shape.reduce{a,b->a*b}
			// First, go over all these
			val index = out.indexToIndexArray(outerIndex)
			if(index.last() == 0 && !started) {
				// Go over all the values in this last item again.
				val indexPrefix = out.indexToIndexArray(outerIndex-1).drop(1)
				for (innerIndex in 0..inputValues[0].shape.last() - 1) {
					val newIndex = indexPrefix.plus(innerIndex)
					out.set(*newIndex.toIntArray(), value=inputValues[0].get(*newIndex.toIntArray()) / accumulator)
				}
				// Reset our values
				started = true
				accumulator = 0f
			} else {
				started = false // We've handled one item.
			}
			accumulator += Math.exp(inputValues[0].get(*index).toDouble()).toFloat()
		}
		return out
	}

	override fun adjointOperation(forwardValues: Array<INDArray>, adjoint: INDArray): Array<INDArray> {
		// If the output probability was [0.2, 0.3, 0.5] and the correct answer was [0, 1, 0],
		// the delta would be [0.2, -0.7, 0.5].
		throw NotImplementedError()
	}
}

class HStackNode(left:Node, right:Node) : Node(IntArray(size=left.shape.size), arrayOf<Node>(left, right)) {

	val splitPoint = left.shape.last()

	// Concatenate two matrices next to one another on the last dimension.
	init {
		assert(left.shape.size == right.shape.size)
		// Copy the dimensions from the left.
		left.shape.mapIndexed { index, value -> this.shape[index] = value }

		// Resize the last index.
		this.shape[this.shape.lastIndex] = left.shape.last()+right.shape.last()
	}

	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		return Nd4j.concat(0, *inputValues)
	}

	override fun adjointOperation(forwardValues: Array<INDArray>, adjoint: INDArray): Array<INDArray> {
		// Split the adjoint.

		return arrayOf(
			adjoint.getColumns(*IntArray(size=forwardValues[0].columns(), init = { a -> a })),
			adjoint.getColumns(*IntArray(size=forwardValues[1].columns(), init = { a -> a+forwardValues[0].columns() }))
		)
	}
}

class ConvolutionNode(input:Node, kernel:Node, stride:Int) : Node(IntArray(size=input.shape.size), arrayOf<Node>(input, kernel)) {
	// Input: Volume of B W1 H1 D1
	// Params: K -> # filters, F -> Spatial extent, S -> Stride, P -> Padding.
	// Output: W2 = (W1 - F + 2P)/S + 1 || H2 is like W2 || D2 = K
	init {
		// Assume we have an in put in R4 and a kernel in R4 that matches.
		val b1 = input.shape[0]

	}

	override fun forwardOperation(vararg inputValues: INDArray): INDArray {
		throw NotImplementedError()
	}

	override fun adjointOperation(forwardValues: Array<INDArray>, adjoint: INDArray): Array<INDArray> {
		throw NotImplementedError()
	}
}
