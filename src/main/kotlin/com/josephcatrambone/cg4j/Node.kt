package com.josephcatrambone.cg4j

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
	abstract fun forwardOperation(vararg inputValues: Tensor): Tensor
	abstract fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor>

	// These are used for serialization.
	// By default, the graph will populate the id, name, shape, and inputs.
	// If a node needs extra data, like a stored value, that has to be put in the extra data.
	open fun extraDataToString(separator:String="|"): String { return "" }
	open fun extraDataFromStringIterator(it:Iterator<String>) {}
}

class InputNode(vararg shape:Int) : Node(shape) {
	//constructor() : this(-1) {} // Allow empty constructor when reloading from disk.  We'll set these later.

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		throw RuntimeException("You should never see this.  The graph should be checking the inputs before this gets called.")
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf()
	}
}

class VariableNode(vararg shape:Int) : Node(shape) {
	var value: Tensor = Tensor.zeros(*shape)

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return value
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf()
	}

	override fun extraDataToString(separator:String):String {
		return value.toString()
	}

	override fun extraDataFromStringIterator(it: Iterator<String>) {
		//super.extraDataFromStringIterator(it)
		this.value = Tensor.fromString(it.next())
	}

}

class AddConstantNode(lhs:Node, var c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].add(c)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
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

class AddNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].add(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint, adjoint)
	}
}

class SubtractNode(lhs:Node, rhs:Node) : Node(lhs.shape, arrayOf<Node>(lhs, rhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].sub(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint, adjoint.neg())
	}
}

class ConstantMultiplyNode(lhs:Node, var c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].mul(c)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
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
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].mul(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(forwardValues[1].mul(adjoint), forwardValues[0].mul(adjoint))
	}
}

class MatrixMultiplyNode : Node {
	// TODO: Figure out how to better use the tensor product here.
	constructor(lhs:Node, rhs:Node) : super(shape = intArrayOf(lhs.shape[0], rhs.shape[1]), inputs = arrayOf<Node>(lhs, rhs)) {

	}

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].mmul(inputValues[1])
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
			// Left adjoint. If C=AB, adj(a) = adj(c)*bT
			adjoint.mmul(forwardValues[1].transpose()),
			// Right adjoint.  adj(b) = aT*adj(c)
			forwardValues[0].transpose().mmul(adjoint)
		)
	}
}

class TanhNode(n:Node) : Node(shape=n.shape, inputs=arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].tanh()
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
				adjoint.mul(Tensor.ones(*adjoint.shape).sub(forwardValues[0].tanh().mul(forwardValues[0].tanh())))
		)
	}
}

class SigmoidNode(n:Node) : Node(shape=n.shape, inputs=arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].elementOperation { x -> 1.0f/(1.0f+Math.exp(-x.toDouble()).toFloat()) }
	}

	// Derivative of sigmoid = s * (1-s)
	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
			adjoint.mul(forwardValues[0].elementOperation { x -> (1.0f/(1.0f+Math.exp(-x.toDouble()).toFloat())) * (1.0f-(1.0f/(1.0f+Math.exp(-x.toDouble()).toFloat()))) })
		)
	}
}

class PowerNode(base:Node, var exponent: Float) : Node(shape=base.shape, inputs=arrayOf<Node>(base)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].pow(exponent)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
			adjoint.mul(forwardValues[0].pow(exponent-1.0f).mul(exponent))
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
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].abs()
	}

	override fun adjointOperation(forwardValues: Array<Tensor>, adjoint: Tensor): Array<Tensor> {
		return arrayOf(
			adjoint.mul(forwardValues[0].sign())
		)
	}
}

// Less rigorous mathematically than a softmax node.  Normalizes the gradient and the forward pass.
class NormalizeNode(n:Node, var axis:Int) : Node(n.shape, arrayOf<Node>(n)) {

	fun normalize(tensor:Tensor, axis:Int): Tensor {
		val output = Tensor.zeros(*tensor.shape)
		for(i in (0..tensor.shape[axis])) {
			// Get this slice, total it up,
			// Don't forget to add this to graph for serialize/deser.
			val st = tensor.getSubtensor(axis, i)
			val low:Float = st.data.min()!!
			val high:Float = st.data.max()!! - low
			output.setSubtensor(axis, i, st.add(-low).mul(1.0f/(1.0e-6f+high)))
		}
		return output
	}

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		// Do slices along the given axis.
		return normalize(inputValues[0], axis)
	}

	override fun adjointOperation(forwardValues: Array<Tensor>, adjoint: Tensor): Array<Tensor> {
		return arrayOf(
				normalize(adjoint, axis)
		)
	}

	override fun extraDataToString(separator:String):String {
		return axis.toString()
	}

	override fun extraDataFromStringIterator(it: Iterator<String>) {
		this.axis = it.next().toInt()
	}
}

class GradientClipNode(n:Node) : Node(n.shape, arrayOf<Node>(n)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0]
	}

	override fun adjointOperation(forwardValues: Array<Tensor>, adjoint: Tensor): Array<Tensor> {
		val magnitude = Math.sqrt(adjoint.data.foldRight(0f, {acc, operand-> acc + operand*operand}).toDouble()).toFloat()
		if(magnitude > 1.0f) { // Magnitude can't be negative.
			return arrayOf(
				adjoint.elementOperation { x -> x/magnitude }
			)
		} else {
			return arrayOf(
				adjoint
			)
		}
	}
}

class SimpleConvolutionNode(input:Node, numFilters:Int, stride:Int, spatialExtent:Int) : Node(
		shape=intArrayOf(),
		inputs=arrayOf<Node>()
) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].abs()
	}

	override fun adjointOperation(forwardValues: Array<Tensor>, adjoint: Tensor): Array<Tensor> {
		return arrayOf(
				adjoint.mul(forwardValues[0].sign())
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

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].abs()
	}

	override fun adjointOperation(forwardValues: Array<Tensor>, adjoint: Tensor): Array<Tensor> {
		return arrayOf(
				adjoint.mul(forwardValues[0].sign())
		)
	}
}
