---
layout: post
title: "Introduction to the Multithreading Problem and the Akka Actor Solution"
date: 2014-12-27
comments: false
categories: 
---

## The Multithreading Problem
Nowadays, computers have multiple execution cores meaning that they can execute multiple tasks at the same time rather than sequentially.  Obviously this makes things much faster but it also presents some new problems.  The term _multithreading_ refers to the process in which multiple threads execute code in the same program simultaneously.  The inherent problem with multithreading lies in the fact that although each thread acts independently, their memory is shared.  Therefore, it is possible for threads to change shared memory values without other threads knowing which can create problems.  Let's use a bank account as an example.  Consider the following code that implements a bank account with ```deposit``` and ```withdraw``` methods.

```scala
class BankAccount{
    
    private var balance = 0
    
    def deposit(amount: Int): Unit = 
        if (amount > 0) balance = balance + amount
        
    def withdraw(amount: Int): Int =
        if (0 < amount && amount <= balance){
            balance = balance - amount
            balance
        } else throw new Error("insufficient funds")

}
```

Here is a scenario where multithreading can cause problems.  Let's say that ```balance```=40 dollars and _thread A_ would like to withdraw 30 dollars. This satisfies both conditions in the _if_ statement so _thread A_ enters the code block where the 30 dollars is subtracted from the balance.  However, before _thread A_ changes the balance, a second thread, _thread B_, wants to withdraw 20 dollars.  Since _thread A_ has not yet changed the balance, _thread B_ also satisfies the _if_ statement and enters the code block where the balance can be changed.  So _thread A_ subtracts $30 from the balance and then _thread B_ subtracts 20 dollars from the balance leaving us with a balance of -10 dollars. Clearly, this is a problem!

Hopefully it is clear that the problem comes from the fact that the shared _balance_ variable can be changed by any thread at any time so no thread can really be sure what value it holds.  One solution is for a thread to be able to reserve the memory values that it will be using so that no other thread can change them.  This is called _locking_.

## Using Locks (Synchronous)
As stated previously, locking tries to solve the multithreading problem by _protecting_ the shared memory value (in this case ```balance```). Scala does this with _synchronization_.  Consider the same code as above but now each method definition is wrapped in ```this.synchronized```.

```scala
class BankAccount{
    
    private var balance = 0
    
    def deposit(amount: Int): Unit = this.synchronized {
        if (amount > 0) balance = balance + amount
    }
        
    def withdraw(amount: Int): Int = this.synchronized {
        if (0 < amount && amount <= balance){
            balance = balance - amount
            balance
        } else throw new Error("insufficient funds")
   }

}
```

Now ```deposit``` and ```withdraw``` each exit as one atomic unit meaning that only one thread can access them at a time (all others are blocked and must wait until the blocking thread is finished).  Now that all methods that can change ```balance``` are synchronized, conflicts such as the ones described above can no longer occur.  Unfortunately, the problem still is not completely solved because synchronization produces a few new problems.  For example, consider another ```BankAccount``` method called ```transfer``` which withdraws money from one account and deposits it into another with the following synchronized code.

```scala
def transfer(from: BankAccount, to: BankAccount, amount: Int): Unit = {
    from.synchronized {
        to.synchronized {
            from.withdraw(amount)
            to.deposit(amount)
        }
    }
}
```

The withdrawal and deposit steps must be synchronized so that no thread can access the balance between withdrawal and deposit (at this point the amount transfered would not be anywhere).  The problem occurs when _thread A_ wants to transfer money from _account A_ to _account B_ at the same time that _thread B_ wants to transfer money from _account B_ to _account A_.  When this happens _thread A_ would lock _account A_ and _thread B_ would lock _account B_ and each thread would wait for the other to release the lock which would take forever!  This is not good and it is called a _dead-lock_ which is a common problem with synchronization.  There are ways of dealing with _dead-locks_ but they are complicated and can make your code difficult to read.  In addition, stopping and starting threads when they become blocked turns out to be very bad for CPU utilization which will make your code run slower.  It would be much better if we could deal with this multithreading problem in such a way that we do not have to use any kind of blocking.  This is what the Akka actor model does.

## Using Actors (Asynchronous)  
Akka actors are fully encapsulated entities.  Changes to their internal state can only be done through passing known messages.  Message passing between actors is one-way and completely asynchronous (i.e. unblocking) so when an actor sends a message it does not have to wait for a reply, it can continue performing other tasks.  If multiple messages are sent to a single actor, it will process them sequentially in a queue (so internally an actor is single-threaded).  If a received message changes an actor's internal state, the change is reflected immediately after the message has been processed.  Therefore, processing one message is the atomic unit of execution (it can never be interrupted).

In terms of our bank account example, let's create a ```BankAccount``` actor that can receive ```Deposit``` and ```Withdraw``` messages.  In Scala, we create an actor by extending the ```Actor``` trait and implementing its ```receive``` method.  We must also define the messages that it can send and recieve in the actor's companion object.  The following is an actor-based ```BankAccount``` implementation.

```scala
object BankAccount {

  case class Deposit(amount: BigInt) {
    require(amount > 0)
  }

  case class Withdraw(amount: BigInt) {
    require(amount > 0)
  }

  case object Done

  case object Failed

}

//Actor that receives messages to perform actions of a bank account
class BankAccount extends Actor {

  import BankAccount._

  var balance = BigInt(0)

  def receive = LoggingReceive {
    //Deposit messages add amount to balance state
    case Deposit(amount) =>
      balance += amount
      sender ! Done

    //Withdraw messages subtract amount from balance state
    case Withdraw(amount) if amount <= balance =>
      balance -= amount
      sender ! Done

    //Any other message would return a failure to the sender
    case _ => sender ! Failed
  }

}
``` 

In the ```BankAccount``` companion object four messages are defined ```Deposit```, ```Withdraw```, ```Done```, and ```Failed```.  In the ```receive``` method in the ```BankAccount``` class defines how to change the ```balance``` variable when either the ```Deposit``` or the ```Withdraw``` method is received.  Once this is finished, a ```Done``` message is passed back to the actor that sent it via the ```sender``` variable (an actor's reference is always tied to the message it sends).  If any other message is received, then a ```Failed``` message is sent back to the sending actor.

But we also want to use an actor to transfer money between accounts.  You may remember that this had the potential to produce a _dead-lock_ when done synchronously. We can avoid this using actors because now blocking is replaced with enqueuing messages.  Let's call this actor ```WireTransfer```.  This actor can receive a ```Transfer``` message which contains three fields - a reference to the sending ```BankAccount``` actor, a reference to the receiving ```BankAccount``` actor, and the amount to be transferred.  When ```WireTransfer``` receives this message, it sends a ```Withdraw``` message (defined within the ```BankAccount``` companion object) to the sending actor, awaits a successful ```Done``` response, then sends a ```Deposit``` message to the recieving actor.  The following code implements the ```WireTransfer``` actor.

```scala
object WireTransfer {

  case class Transfer(from: ActorRef, to: ActorRef, amount: BigInt)

  case object Done

  case object Failed

}

//actor implementing the actions of a wire transfer between two bank account actors
class WireTransfer extends Actor {

  import WireTransfer._

  def receive = LoggingReceive {
    //If Transfer message is received, send withdraw message to 'from' and wait for reply
    case Transfer(from, to, amount) =>
      from ! BankAccount.Withdraw(amount)
      context.become(awaitFrom(to, amount, sender))
  }

  //If Withdraw was successful, send deposit to other bank account actor, or else give them a failure message
  def awaitFrom(to: ActorRef, amount: BigInt, customer: ActorRef): Receive = LoggingReceive {
    case BankAccount.Done =>
      to ! BankAccount.Deposit(amount)
      context.become(awaitTo(customer))
    case BankAccount.Failed =>
      customer ! Failed
      context.stop(self)
  }

  //If deposit was successful, send 'Done' to original actor that sent Transfer message
  def awaitTo(customer: ActorRef): Receive = LoggingReceive {
    case BankAccount.Done =>
      customer ! Done
      context.stop(self)
  }
}
```

When the actor first receives the ```Transfer``` message, it sends a ```Withdraw``` message to the ```BankAccount``` actor referenced in the ```from``` field of the message.  Then the actor must wait until a ```Done``` message is received (it does this with the ```context.become()``` method) at which point it sends a ```Deposit``` message to the other ```BankAccount``` actor.

We can also test this transfer process by creating a new actor that creates two ```BankAccount``` actors and a ```WireTransfer``` actor and then sends a ```Transfer``` message to the ```WireTransfer``` actor which references both of the ```BankAccount``` actors.  But before the ```Transfer``` message is sent, the actor must deposit some money in the first ```BankAccount``` actor so that there is some money available to transfer.  Let's call this actor ```TransferMain``` and here is its implementation.

```scala
class TransferMain extends Actor {

  //First create two BankAccount actors
  val accountA = context.actorOf(Props[BankAccount], "accountA")
  val accountB = context.actorOf(Props[BankAccount], "accountB")

  //send a deposit message to accountA
  accountA ! BankAccount.Deposit(100)

  //If a 'Done' message is received back, call a transfer function
  def receive = LoggingReceive {
    case BankAccount.Done => transfer(70)
  }

  //transfer function creates a transacton actor and sends a 'Transfer' message to it between
  //accountA and accountB for the specified amount.
  def transfer(amount: BigInt): Unit = {

    val transaction = context.actorOf(Props[WireTransfer], "transfer")

    transaction ! WireTransfer.Transfer(accountA, accountB, amount)

    context.become(LoggingReceive {
      case WireTransfer.Done =>
        println("successs")
        context.stop(self)
      case WireTransfer.Failed =>
        println("failed")
        context.stop(self)
    })

  }
}
``` 

As you can see from the above code, 100 dollars is deposted in ```accountA``` via the ```Deposit``` message.  Then, when the ```Done``` message is recieved, the ```transfer``` function is called with an argument of 70 dollars.  Inside the ```transfer``` function, a ```WireTransfer``` actor is created and a ```Transfer``` message is sent to it with the appropriate arguments.  It then waits for either a ```Done``` or ```Failed``` message in return.

Hopefully this blog post has shed some light on the multithreading problem and how the Akka actor model tries to solve it.

## References
* This bank account example was taken from the [Principles of Reactive Programming](https://www.coursera.org/course/reactive) Coursera course.
* The full code snippets can be found [on github](https://github.com/alexminnaar/Scala_Code_Snippets/tree/master/src/main/scala/BankAccount).