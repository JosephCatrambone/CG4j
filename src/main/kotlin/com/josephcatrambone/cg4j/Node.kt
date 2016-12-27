package com.josephcatrambone.cg4j

/**
 * Created by jcatrambone on 12/22/16.
 */
abstract class Node(var shape:IntArray, var inputs:Array<Node>) {
	var id: Int = -1 // When we serialize, we write these IDs instead of recursively serualizing objects.
	var name:String = ""
	abstract fun forwardOperation(vararg inputValues: Tensor): Tensor
	abstract fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor>

	override fun toString(): String {
		val sb = StringBuilder()
		// Class Name.  This will get stripped off by the parent class.
		sb.append(this.javaClass.canonicalName)
		sb.append('|')
		// ID
		sb.append(this.id)
		sb.append('|')
		// Name
		sb.append(this.name)
		sb.append('|')
		// Shape
		sb.append(this.shape.joinToString(separator = ","))
		sb.append('|')
		// Inputs
		for(n in this.inputs) {
			sb.append(n.id)
			sb.append(',')
		}
		sb.append('|')
		sb.append(extrasToString('|'))
		return sb.toString()
	}

	fun fromString(g: Graph, line:String) { // Need Graph so we can do the 'Get Node' thing.
		val tokens = line.splitToSequence('|').iterator()
		this.id = tokens.next().toInt()
		this.name = tokens.next()
		this.shape = tokens.next().split(',').map{ s -> s.toInt() }.toIntArray()
		this.inputs = tokens.next().split(',').map{ s -> g.nodes[s.toInt()] }.toTypedArray<Node>()

		extrasFromString(tokens)
	}

	fun extrasToString(delimiter:Char='|'): String { return "" }
	fun extrasFromString(iter:Iterator<String>) {}
}

class InputNode(vararg shape:Int) : Node(shape, inputs=arrayOf<Node>()) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		throw RuntimeException("You should never see this.  The graph should be checking the inputs before this gets called.")
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf()
	}
}

class VariableNode(vararg shape:Int) : Node(shape, inputs=arrayOf<Node>()) {
	var value: Tensor = Tensor.zeros(*shape)

	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return value
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf()
	}
}

class AddConstantNode(lhs:Node, val c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].add(c)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint)
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

class ConstantMultiplyNode(lhs:Node, val c:Float) : Node(lhs.shape, arrayOf<Node>(lhs)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].mul(c)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(adjoint.mul(c))
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

class PowerNode(base:Node, val exponent: Float) : Node(shape=base.shape, inputs=arrayOf<Node>(base)) {
	override fun forwardOperation(vararg inputValues: Tensor): Tensor {
		return inputValues[0].pow(exponent)
	}

	override fun adjointOperation(forwardValues:Array<Tensor>, adjoint:Tensor): Array<Tensor> {
		return arrayOf(
			adjoint.mul(forwardValues[0].pow(exponent-1.0f).mul(exponent))
		)
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
