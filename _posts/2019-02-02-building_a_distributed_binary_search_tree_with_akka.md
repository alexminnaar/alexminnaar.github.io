---
layout: post
title: "Building a Distributed Binary Search Tree with Akka"
date: 2015-01-05
comments: false
categories: 
---

In this blog post I will descibe an interesting Akka mini-project that I came across which helped me gain a deeper understanding of Akka's asynchronous actor model.  In this project we use Akka to build a distributed binary search tree where each node in the tree is an actor which allows it to be a completely asynchronous, concurrent, and distributed version of the traditional data structure.  But before we get into the Akka stuff, it would be helpful to remind ourselves of some of the basic properties of a binary search tree.

## Binary Search Tree Basics

A binary search tree is a tree-based data structure with the following properites

<ol style="margin-left: 20px">
  <li style="font-size:19px">Each node in the tree stores an element and can have at most two child nodes.</li>
  <li style="font-size:19px">The tree does not contain any duplicate values.</li>
  <li style="font-size:19px">Elements in a node's left subtree are strictly smaller than the node's element.</li>
  <li style="font-size:19px">Elements in a node's right subtree are strictly greater than the node's element.</li>
</ol>


<div style="text-align:center">

<img src="{{site.baseurl}}/assets/Binary_search_tree.svg">
</div>


There are also three main tasks a binary search tree can perform.

<ol style="margin-left: 20px">
  <li style="font-size:19px"><b>Contains</b>: Determine if a node containing a particular element exists in the tree by performing a binary tree search.  This is done by starting at the root node and recursively searching the tree by selecting the left or right subtree based on the node's value and the value that we are searching for.  We know that the value does not exist in the tree if we reach an external node and have not yet found it.</li>
  <li style="font-size:19px"><b>Insert</b>: Insert a new node in the correct place in the tree given its value.  This is also done with a tree search.  Assuming the element does not already exist in the tree, we search for the value that we wish to insert until we arrive at an external node at which point we add the new node as its right or left child depending on its value.</li>
  <li style="font-size:19px"><b>Remove</b>: Remove an element from the tree and rearrange the remaining nodes in order to keep the desired structure.  If the node you wish to delete is an external node you simply remove it, however if it has children it is more complicated.  One way to deal with this is to identify its in-order predecessor in the left subtree.  This is the greatest element in the left subtree which can be found by recursively selecting the right child within this subtree (in other words it is the right-most element in the left subtree).  This in-order predecessor is then removed (it is an external node so this is simple) and is used to replace the node that is to be deleted.  The opposite procedure would work as well (i.e. replacing the node with its in-order successor in the right subtree).</li>
</ol>


Before we create an Akka application that implements a binary search tree let's briefly review the basics of the actor model.

## Actor Model Basics 
  
  Actors are completely incapsulated, asynchronous entities that are each designed to perform a specific task.  The only way that actors can communicate with each other is through message passing.  Message passing is asynchronous meaning an actor can send a message and then immediately continue performing other tasks (it doesn't have to wait for a response).  When a message is sent to an actor, it is put in a queue and the actor performs the tasks corresponding to each message sequentially, therefore actors are themselves single-threaded.  Furthermore, due to these properties, actors can be distributed in a cluster with essentially the same code as if they were on the same machine which makes things very convenient.
  
Now let's start to build our Akka application.
  
## Tree Nodes as Actors
We will create one main actor called ```BinaryTreeSet``` which receives the ```Contains```, ```Insert```, and ```Remove``` messages for the entire tree.  As stated previously, each node of the tree will also be an actor which we will call ```BinaryTreeNode```.  Actors usually incapsulate an immutable state.  The ```BinaryTreeSet``` actor's state contains the root node of the tree which is a ```BinaryTreeNode``` actor.  Each ```BinaryTreeNode``` actor's state contains the value that the node holds as well as references to its two children (which are also ```BinaryTreeNode``` actors).

Let's first focus on the ```BinaryTreeSet``` actor.  An actor class extends Akka's ```Actor``` trait and its messages are customarily defined in its companion object as case classes. The following code implements this companion object.

```scala
object BinaryTreeSet {

  trait Operation {
    def requester: ActorRef
    def id: Int
    def elem: Int
  }

  trait OperationReply {
    def id: Int
  }
  
  case class Insert(requester: ActorRef, id: Int, elem: Int) extends Operation
  case class Contains(requester: ActorRef, id: Int, elem: Int) extends Operation
  case class Remove(requester: ActorRef, id: Int, elem: Int) extends Operation
  case class OperationFinished(id: Int) extends OperationReply
}
```

The ```Insert```, ```Contains```, and ```Remove``` messages each have three fields

<ul style="margin-left: 20px">
  <li style="font-size:19px"><b>requester</b> which is a reference to the actor that sent the request.</li>
  <li style="font-size:19px"><b>id</b> which is a unique id for the message.</li>
  <li style="font-size:19px"><b>elem</b> which is the value to be inserted (<b>Insert</b>), removed (<b>Remove</b>) or searched for (<b>Contains</b>).</li>
</ul>

There is also an ```OperationFinished``` message which is sent back to the ```BinaryTreeSet``` actor when the operation specified by the ```id``` field is finished.  

Now let's look at the ```BinaryTreeSet``` actor class.  

```scala
class BinaryTreeSet extends Actor {
  import BinaryTreeSet._
  import BinaryTreeNode._

  def createRoot: ActorRef = context.actorOf(BinaryTreeNode.props(0, initiallyRemoved = true))

  var root = createRoot

  def receive = normal
  
  val normal: Receive = {
    case operation: Operation => root ! operation
  }
}
```

Here the tree's root node actor ```root``` is created using the ```context.actorOf``` method.  The ```Receive``` method accepts ```Operation``` messages and sends them to ```root```.  Essentially, all messages are sent to the root node and are executed within nodes of the tree which are ```BinaryTreeNode``` actors.  So let's look at the ```BinaryTreeNode``` implementation.

```scala
object BinaryTreeNode {
  trait Position

  case object Left extends Position
  case object Right extends Position

  def props(elem: Int, initiallyRemoved: Boolean) = Props(classOf[BinaryTreeNode],  elem, initiallyRemoved)
}
class BinaryTreeNode(val elem: Int, initiallyRemoved: Boolean) extends Actor {
  import BinaryTreeNode._
  import BinaryTreeSet._

  var subtrees = Map[Position, ActorRef]()
  var removed = initiallyRemoved

  def receive = normal

  val normal: Receive = {...}
}
```

Where the ```subtrees``` map holds the node's left and right children which are themselves ```BinaryTreeNode``` actors. The ```Receive``` method is left blank so that we can look at the ```Operation``` message implementations in more detail.  Let's do this now.

### Contains Messages
As mentioned previously, a binary tree can be searched by recursively selecting each child depending on the value of the current node and the value of the element that is being searched for until either the element is found or an external node is reached.  The child to select can be found using this simple function

```scala
def childToVisit(elemToFind: Int): Position = {
    if (elemToFind > elem) Right
    else Left
}
```

We can now implement the functionality to process ```Contains``` messages as follows.

```scala
case Contains(requester, id, elemToFind) => {

      if (elem != elemToFind || (elem == elemToFind && removed)) {

        val child = childToVisit(elemToFind)

        if (subtrees.contains(child)) {
          subtrees(child) ! Contains(requester, id, elemToFind)
        }
        else {
          requester ! ContainsResult(id, false)
        }
      }
      else {
        requester ! ContainsResult(id, true)
      }
}
``` 

Basically, if the desired value is in the current node and hasn't been removed then the ```ContainsResult(id, true)``` message is sent back to the requester.  If it is not and it is not an external node, then the same message is sent to the correct child.  If it is an external node then the element does not exist in the tree and a ```ContainsResult(id, false)``` is sent back to the requester.

### Insert Messages
```Insert``` messages can be handled in a similar way.  Again we search the tree and when we get to an external node we create a new ```BinaryTreeNodeActor``` that holds the element to insert and add this node to the external node's ```subtree``` map.

```scala
case Insert(requester, id, elemToInsert) => {

      if (elem != elemToInsert || (elem == elemToInsert && removed)) {

        val child = childToVisit(elemToInsert)

        if (subtrees.contains(child)) {
          subtrees(child) ! Insert(requester, id, elemToInsert)
        }
        else {
          subtrees += (child -> context.actorOf(BinaryTreeNode.props(elemToInsert, false)))
          requester ! OperationFinished(id)
        }
      }
      else {
        requester ! OperationFinished(id)
      }
}
```


### Remove Messages  
Remove messages are more difficult to deal with.  We will not be implementing the same removal procedure as described in _Binary Search Tree Basics_.  Unlike node search and insertion, node removal results in a tree restructuring.  This is problematic in aysnchronous applications.  For example, what if  a removal causes a tree restructuring that occurs while other messages are still being processed and coming in? Synchronization is required for tree restructuring which does not fit within the actor model.  For this reason, we will handle removal by giving each node a ```removed``` flag that indicates if the node has been removed.  This way, removal occurs by simply setting the ```removed``` flag to ```true```.

```scala
case Remove(requester, id, elemToRemove) => {

      if (elem != elemToRemove || (elem==elemToRemove && removed)) {

        val child = childToVisit(elemToRemove)

        if (subtrees.contains(child)) {
          subtrees(child) ! Remove(requester, id, elemToRemove)
        }
        else {
          requester ! OperationFinished(id)
        }
      }
      else {
        removed = true
        requester ! OperationFinished(id)
      }
}
```

This makes removal asynchronous and much simpler however it also means that we will be accumulating nodes that have been removed which could become problematic in terms of memory.  We will deal with this by introducing a new type of message called ```GC``` (for _Garbage Collection_) that the main ```BinaryTreeSet``` actor can receive.  When this message is received, all of the nodes in the tree that haven't been removed (i.e. where ```removed = false```) are copied and inserted into a new tree (with a new root node) which results in a new tree where the nodes that have been removed are _actually_ removed.  Also, when the copy is completed, all of the actors in the old tree are stopped.  There will also be a ```CopyTo``` message which holds the root node of the new tree as a field.  This message is recursively sent to each node in the old tree and if its ```removed``` flag is ```false``` then it is inserted into the new tree. One more thing we must deal with is what to do with messages that come in while garbage collection is taking place.  We will deal with this by enqueuing these messages and then begin processing them once garbage collection has completed.

So once the ```GC``` message is received by the ```BinaryTreeSet``` actor, it enters into a new context where it waits for the new tree to be copied while also enqueuing other messages that are received during this time.

```scala
case GC => {
      val newRoot = createRoot
      root ! CopyTo(newRoot)
      context.become(garbageCollecting(newRoot))
    }
```

```scala
  def garbageCollecting(newRoot: ActorRef): Receive = {

    case operation: Operation => pendingQueue.enqueue(operation)

    case CopyFinished => {
      root ! PoisonPill
      val newRoot = createRoot
      root = newRoot

      pendingQueue.map(root ! _)
      pendingQueue = Queue.empty

      context.become(normal)
    }
}
```

Then once it receives the ```CopyFinished``` message, it executes all of the messages that are in the queue and  returns to its normal context.  For the ```BinaryTreeNode``` actors, the ```CopyTo``` message is handled as follows.

```scala
case CopyTo(newRoot) => {
      if (!removed){
        newRoot ! Insert(self, 0, elem)
      }
      
      subtrees.values foreach (_ ! CopyTo(newRoot))

      if (removed && subtrees.isEmpty){
        sender ! CopyFinished
      }
      else{
        context.become(copying(subtrees.values.toSet, insertConfirmed = removed, sender))
      }
}
```

i.e. if the node has not been removed, insert it into the new tree.  Alternatively, if it has been removed and it is an external node, then there is nothing left to copy and the ```CopyFinished``` message is sent back to the sending node.  Once a node sends the ```CopyTo``` message to its children, it enters a ```copying``` context in which it waits for each of its children to return a ```CopyFinished``` message, at which point the node itself returns a ```CopyFinished``` message to its parent until eventually the ```BinaryTreeSet``` actor (the actor that initially sent the ```CopyTo``` message) receives a ```CopyFinished``` message and we know that all nodes have been copied. The ```copying``` context is shown below.

```scala
def copying(expected: Set[ActorRef], insertConfirmed: Boolean, originator: ActorRef): Receive = {
    case OperationFinished(_) =>
      if (expected.isEmpty) {
        originator ! CopyFinished
        context.become(normal)
      } else {
        context.become(copying(expected, insertConfirmed = true, originator))
      }
    case CopyFinished =>
      val newExpected = expected - sender
      if (newExpected.isEmpty && insertConfirmed) {
        originator ! CopyFinished
        context.become(normal)
      } else {
        context.become(copying(newExpected, insertConfirmed, originator))
      }
}
```

And that is the basic functionality of the distributed binary search tree.  The full code is available on [this github repo](https://github.com/alexminnaar/ActorBinaryTree).  Hopefully this blog post has shed some light on the Akka actor model and how it can be used to build concurrent, distributed applications like this one.

## References
 * [The Akka Homepage](http://akka.io/)
 * [Principles of Reactive Programming Coursera Course](https://www.coursera.org/course/reactive)
 * [GitHub repository](https://github.com/alexminnaar/ActorBinaryTree) for the code used in this post.